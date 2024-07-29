import json
import os
import re
from collections import Counter
import multiprocessing as mp
from itertools import chain
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ID2LABEL = {
    "publaynet": {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"},
    "scipostlayout-max50": {
        0: 'Title',
        1: 'Author Info',
        2: 'Section',
        3: 'List',
        4: 'Text',
        5: 'Caption',
        6: 'Figure',
        7: 'Table',
        8: 'Unknown'
    },
    "rico": {
        1: "text",
        2: "image",
        3: "icon",
        4: "list item",
        5: "text button",
        6: "toolbar",
        7: "web view",
        8: "input",
        9: "card",
        10: "advertisement",
        11: "background image",
        12: "drawer",
        13: "radio button",
        14: "checkbox",
        15: "multi-tab",
        16: "pager indicator",
        17: "modal",
        18: "on/off switch",
        19: "slider",
        20: "map view",
        21: "button bar",
        22: "video",
        23: "bottom navigation",
        24: "number stepper",
        25: "date picker",
    },
    "posterlayout": {1: "text", 2: "logo", 3: "underlay"},
    "webui": {
        0: "text",
        1: "link",
        2: "button",
        3: "title",
        4: "description",
        5: "image",
        6: "background",
        7: "logo",
        8: "icon",
        9: "input",
    },
}


CANVAS_SIZE = {
    "rico": (90, 160),
    "publaynet": (120, 160),
    "scipostlayout-max50": (120, 160),
    "posterlayout": (102, 150),
    "webui": (120, 120),
}


RAW_DATA_PATH = lambda x: os.path.join(os.path.dirname(__file__), f"/scipostlayout/code/LayoutPrompter/datasets/{x}/raw")


LAYOUT_DOMAIN = {
    "rico": "android",
    "publaynet": "document",
    "posterlayout": "poster",
    "webui": "web",
    "scipostlayout-max50": "document",
}


def clean_text(text: str, remove_summary: bool = False):
    if remove_summary:
        text = re.sub(r"#.*?#", "", text)
    text = text.replace("[#]", " ")
    text = text.replace("#", " ")
    text = text.replace("\n", " ")
    text = text.replace(",", ", ")
    text = text.replace(".", ". ").strip()
    text = re.sub(r"[ ]+", " ", text)
    return text


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_pt(filename):
    with open(filename, "rb") as f:
        return torch.load(f)


def write_pt(filename, obj):
    with open(filename, "wb") as f:
        torch.save(obj, f)


def convert_ltwh_to_ltrb(bbox):
    if len(bbox.size()) == 1:
        l, t, w, h = bbox
        r = l + w
        b = t + h
        return l, t, r, b
    l, t, w, h = decapulate(bbox)
    r = l + w
    b = t + h
    return torch.stack([l, t, r, b], axis=-1)


def convert_ltrb_to_ltwh(bbox):
    if len(bbox.size()) == 1:
        l, t, r, b = bbox
        w = r - l
        h = b - t
        return l, t, w, h
    l, t, r, b = decapulate(bbox)
    w = r - l
    h = b - t
    return torch.stack([l, t, w, h], axis=-1)


def decapulate(bbox):
    if len(bbox.size()) == 2:
        x1, y1, x2, y2 = bbox.T
    else:
        x1, y1, x2, y2 = bbox.permute(2, 0, 1)
    return x1, y1, x2, y2


def detect_size_relation(b1, b2):
    REL_SIZE_ALPHA = 0.1
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    a1_sm = (1 - REL_SIZE_ALPHA) * a1
    a1_lg = (1 + REL_SIZE_ALPHA) * a1

    if a2 <= a1_sm:
        return "smaller"

    if a1_sm < a2 and a2 < a1_lg:
        return "equal"

    if a1_lg <= a2:
        return "larger"

    raise RuntimeError(b1, b2)


def detect_loc_relation(b1, b2, canvas=False):
    if canvas:
        yc = b2[1] + b2[3] / 2
        y_sm, y_lg = 1.0 / 3, 2.0 / 3

        if yc <= y_sm:
            return "top"

        if y_sm < yc and yc < y_lg:
            return "center"

        if y_lg <= yc:
            return "bottom"

    else:
        l1, t1, r1, b1 = convert_ltwh_to_ltrb(b1)
        l2, t2, r2, b2 = convert_ltwh_to_ltrb(b2)

        if b2 <= t1:
            return "top"

        if b1 <= t2:
            return "bottom"

        if t1 < b2 and t2 < b1:
            if r2 <= l1:
                return "left"

            if r1 <= l2:
                return "right"

            if l1 < r2 and l2 < r1:
                return "center"

    raise RuntimeError(b1, b2, canvas)


def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = bbox.unsqueeze(-1)
    l2, t2, r2, b2 = bbox.unsqueeze(-2)
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = ai / a1
    ar = torch.from_numpy(np.nan_to_num(ar.numpy()))
    score = torch.from_numpy(
        np.nan_to_num((ar.sum(dim=(1, 2)) / mask.float().sum(-1)).numpy())
    )
    return (score).mean().item()


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = bbox
    xc = (xr + xl) / 2
    yc = (yt + yb) / 2
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)

    X = -torch.log(1 - X)
    score = torch.from_numpy(np.nan_to_num((X.sum(-1) / mask.float().sum(-1)))).numpy()
    return (score).mean().item()


def compute_iou(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = box_1.T
    l2, t2, r2, b2 = box_2.T
    w1 = lib.where((r1 - l1) == 0, 1e-7, r1 - l1)
    h1 = lib.where((b1 - t1) == 0, 1e-7, b1 - t1)
    w2 = lib.where((r2 - l2) == 0, 1e-7, r2 - l2)
    h2 = lib.where((b2 - t2) == 0, 1e-7, b2 - t2)

    a1, a2 = w1 * h1, w2 * h2

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    return iou


def __compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([
        __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
        for i, j in zip(ii, jj)
    ]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list):
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(
        labels_1: torch.Tensor,
        bboxes_1: torch.Tensor,
        labels_2: List[torch.Tensor],
        bboxes_2: List[torch.Tensor],
        n_jobs=None,
    ):
    layouts_1, layouts_2 = [(bboxes_1.cpu().numpy(), labels_1.cpu().numpy())], [(bboxes_2.cpu().numpy(), labels_2.cpu().numpy())]
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    scores = [__compute_maximum_iou(a) for a in args]
    scores = np.asarray(list(chain.from_iterable(scores)))
    if len(scores) == 0:
        return 0.0
    else:
        return scores.mean().item()


def labels_similarity(labels_1, labels_2):
    def _intersection(labels_1, labels_2):
        cnt = 0
        x = Counter(labels_1)
        y = Counter(labels_2)
        for k in x:
            if k in y:
                cnt += 2 * min(x[k], y[k])
        return cnt

    def _union(labels_1, labels_2):
        return len(labels_1) + len(labels_2)

    if isinstance(labels_1, torch.Tensor):
        labels_1 = labels_1.tolist()
    if isinstance(labels_2, torch.Tensor):
        labels_2 = labels_2.tolist()
    return _intersection(labels_1, labels_2) / _union(labels_1, labels_2)


def bboxes_similarity(labels_1, bboxes_1, labels_2, bboxes_2, times=2):
    """
    bboxes_1: M x 4
    bboxes_2: N x 4
    distance: M x N
    """
    distance = torch.cdist(bboxes_1, bboxes_2) * times
    distance = torch.pow(0.5, distance)
    mask = labels_1.unsqueeze(-1) == labels_2.unsqueeze(0)
    distance = distance * mask
    row_ind, col_ind = linear_sum_assignment(-distance)
    return distance[row_ind, col_ind].sum().item() / len(row_ind)


def labels_bboxes_similarity(
    labels_1, bboxes_1, labels_2, bboxes_2, labels_weight, bboxes_weight
):
    labels_sim = labels_similarity(labels_1, labels_2)
    bboxes_sim = bboxes_similarity(labels_1, bboxes_1, labels_2, bboxes_2)
    return labels_weight * labels_sim + bboxes_weight * bboxes_sim


def to_dense_batch(batch):
    '''
    padding a batch of data with value 0
    '''
    unsqueeze_flag = False
    if batch[-1].dim() == 1:
        unsqueeze_flag = True
        batch = [data.unsqueeze(-1) for data in batch]
    lens = [len(data) for data in batch]
    max_lens = max(lens)

    fill_size = batch[-1][-1].size()
    fill_unit = torch.zeros(fill_size, dtype=batch[-1].dtype)

    out = torch.cat((batch[0], fill_unit.repeat((max_lens - lens[0]), 1)),
                    dim=0).unsqueeze(0)
    for i in range(1, len(lens)):
        out = torch.cat((out, (torch.cat(
            (batch[i], fill_unit.repeat(
                (max_lens - lens[i]), 1)), dim=0).unsqueeze(0))),
                        dim=0)
    if unsqueeze_flag:
        out = out.squeeze(-1)

    mask = [[True] * i + [False] * (max_lens - i) for i in lens]
    mask = torch.from_numpy(np.array(mask))

    return out, mask

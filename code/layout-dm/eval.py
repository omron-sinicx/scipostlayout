import argparse
import os
import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from fsspec.core import url_to_fs
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from trainer.data.util import loader_to_list, sparse_to_dense
from trainer.fid.model import load_fidnet_v3
from trainer.global_configs import FID_WEIGHT_DIR
from trainer.helpers.metric import Layout
from trainer.helpers.util import set_seed

import torch
from eval_utils.fid import LayoutFID
from eval_utils.utils import convert_ltwh_to_ltrb, convert_ltrb_to_ltwh, compute_overlap, compute_alignment, compute_maximum_iou
from PIL import Image, ImageDraw
import seaborn as sns
import torchvision.utils as vutils
import torchvision.transforms as T

def preprocess(layouts: List[Layout], max_len: int, device: torch.device):
    layout = defaultdict(list)
    for (b, l) in layouts:
        pad_len = max_len - l.shape[0]
        bbox = torch.tensor(
            np.concatenate([b, np.full((pad_len, 4), 0.0)], axis=0),
            dtype=torch.float,
        )
        layout["bbox"].append(bbox)
        label = torch.tensor(
            np.concatenate([l, np.full((pad_len,), 0)], axis=0),
            dtype=torch.long,
        )
        layout["label"].append(label)
        mask = torch.tensor(
            [True for _ in range(l.shape[0])] + [False for _ in range(pad_len)]
        )
        layout["mask"].append(mask)
    bbox = torch.stack(layout["bbox"], dim=0).to(device)
    label = torch.stack(layout["label"], dim=0).to(device)
    mask = torch.stack(layout["mask"], dim=0).to(device)
    padding_mask = ~mask
    return bbox, label, padding_mask, mask


def print_scores(scores: Dict, test_cfg: argparse.Namespace, train_cfg: DictConfig):
    scores = {k: scores[k] for k in sorted(scores)}
    job_name = train_cfg.job_dir.split("/")[-1]
    model_name = train_cfg.model._target_.split(".")[-1]
    cond = test_cfg.cond

    if "num_timesteps" in test_cfg:
        step = test_cfg.num_timesteps
    else:
        step = train_cfg.sampling.get("num_timesteps", None)

    option = ""
    header = ["job_name", "model_name", "cond", "step", "option"]
    data = [job_name, model_name, cond, step, option]

    tex = ""
    for k, v in scores.items():
        # if k == "Alignment" or k == "Overlap" or "Violation" in k:
        #     v = [_v * 100 for _v in v]
        mean, std = np.mean(v), np.std(v)
        stdp = std * 100.0 / mean
        print(f"\t{k}: {mean:.4f} ({stdp:.4f}%)")
        tex += f"& {mean:.4f}\\std{{{stdp:.1f}}}\% "

        header.extend([f"{k}-mean", f"{k}-std"])
        data.extend([mean, std])

    print(tex + "\\\\")

    print(",".join(header))
    print(",".join([str(d) for d in data]))


def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = bbox
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color,
                       fill=c_fill)
    return img


def save_image(batch_boxes, batch_labels, batch_mask,
               dataset_colors, out_path, canvas_size=(60, 40),
               nrow=None):
    # batch_boxes: [B, N, 4]
    # batch_labels: [B, N]
    # batch_mask: [B, N]

    imgs = []
    B = batch_boxes.size(0)
    to_tensor = T.ToTensor()
    for i in range(B):
        mask_i = batch_mask[i]
        boxes = batch_boxes[i][mask_i]
        labels = batch_labels[i][mask_i]
        img = convert_layout_to_image(boxes, labels,
                                      dataset_colors,
                                      canvas_size)
        imgs.append(to_tensor(img))
    image = torch.stack(imgs)

    if nrow is None:
        nrow = int(np.ceil(np.sqrt(B)))

    vutils.save_image(image, out_path, normalize=False, nrow=nrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=str, default="tmp/results")
    parser.add_argument(
        "--compute_real",
        action="store_true",
        help="compute some metric between validation and test subset",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="number of samples used for evaluating unconditional generation",
    )
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()
    set_seed(0)

    fs, _ = url_to_fs(args.result_dir)
    pkl_paths = [p for p in fs.ls(args.result_dir) if p.split(".")[-1] == "pkl"]
    with fs.open(pkl_paths[0], "rb") as file_obj:
        meta = pickle.load(file_obj)
        train_cfg, test_cfg = meta["train_cfg"], meta["test_cfg"]
        assert test_cfg.num_run == 1

    train_cfg.data.num_workers = 5

    kwargs = {
        "batch_size": args.batch_size,
        "num_workers": train_cfg.data.num_workers,
        "pin_memory": True,
        "shuffle": False,
    }

    if test_cfg.get("is_validation", False):
        split_main, split_sub = "val", "test"
    else:
        split_main, split_sub = "test", "val"

    main_dataset = instantiate(train_cfg.dataset)(split=split_main, transform=None)
    if test_cfg.get("debug_num_samples", -1) > 0:
        main_dataset = main_dataset[: test_cfg.debug_num_samples]
    main_dataloader = DataLoader(main_dataset, **kwargs)
    layouts_main = loader_to_list(main_dataloader)

    if args.compute_real:
        sub_dataset = instantiate(train_cfg.dataset)(split=split_sub, transform=None)
        if test_cfg.cond == "unconditional":
            sub_dataset = sub_dataset[: args.num_samples]
        sub_dataloader = DataLoader(sub_dataset, **kwargs)
        layouts_sub = loader_to_list(sub_dataloader)

    num_classes = len(main_dataset.labels)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # fid_model = load_fidnet_v3(main_dataset, FID_WEIGHT_DIR, device)

    scores_all = defaultdict(list)
    feats_1 = []
    batch_metrics = defaultdict(float)
    for i, batch in enumerate(main_dataloader):
        bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
        # with torch.set_grad_enabled(False):
        #     feat = fid_model.extract_features(bbox, label, padding_mask)
        # feats_1.append(feat.cpu())
        # save_image(bbox, label, mask, main_dataset.colors, f"dummy.png")

        if args.compute_real:
            for k, v in compute_alignment(bbox.cpu(), mask.cpu()).items():
                batch_metrics[k] += v.sum().item()
            for k, v in compute_overlap(bbox.cpu(), mask.cpu()).items():
                batch_metrics[k] += v.sum().item()

    if args.compute_real:
        scores_real = defaultdict(list)
        for k, v in batch_metrics.items():
            scores_real.update({k: v / len(main_dataset)})

    # compute metrics between real val and test dataset
    if args.compute_real:
        feats_1_another = []
        for batch in sub_dataloader:
            bbox, label, padding_mask, mask = sparse_to_dense(batch, device)
            with torch.set_grad_enabled(False):
                feat = fid_model.extract_features(bbox, label, padding_mask)
            feats_1_another.append(feat.cpu())

        scores_real.update(compute_generative_model_scores(feats_1, feats_1_another))
        scores_real.update(compute_average_iou(layouts_sub))
        if test_cfg.cond != "unconditional":
            scores_real["maximum_iou"] = compute_maximum_iou(layouts_main, layouts_sub)
            scores_real["DocSim"] = compute_docsim(layouts_main, layouts_main)

        # regard as the result of single run
        scores_real = {k: [v] for (k, v) in scores_real.items()}
        print()
        print("\nReal data:")
        print_scores(scores_real, test_cfg, train_cfg)

    # # compute scores for each run
    # for pkl_path in pkl_paths:
    #     feats_2 = []
    #     batch_metrics = defaultdict(float)

    #     with fs.open(pkl_path, "rb") as file_obj:
    #         x = pickle.load(file_obj)
    #     generated = x["results"]

    #     print("len of generated,", len(generated))
    #     for i in range(0, len(generated), args.batch_size):
    #         i_end = min(i + args.batch_size, len(generated))
    #         batch = generated[i:i_end]
    #         max_len = max(len(g[-1]) for g in batch)

    #         bbox, label, padding_mask, mask = preprocess(batch, max_len, device)
    #         with torch.set_grad_enabled(False):
    #             feat = fid_model.extract_features(bbox, label, padding_mask)
    #         feats_2.append(feat.cpu())

    #         for k, v in compute_alignment(bbox, mask).items():
    #             batch_metrics[k] += v.sum().item()
    #         for k, v in compute_overlap(bbox, mask).items():
    #             batch_metrics[k] += v.sum().item()

    #     scores = {}
    #     for k, v in batch_metrics.items():
    #         scores[k] = v / len(generated)
    #     scores.update(compute_average_iou(generated))
    #     scores.update(compute_generative_model_scores(feats_1, feats_2))
    #     if test_cfg.cond != "unconditional":
    #         scores["maximum_iou"] = compute_maximum_iou(layouts_main, generated)
    #         scores["DocSim"] = compute_docsim(layouts_main, generated)

    #     for k, v in scores.items():
    #         scores_all[k].append(v)

    # print_scores(scores_all, test_cfg, train_cfg)
    # print()

    # compute scores with layoutformer++ standard
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    for pkl_path in pkl_paths:
        fid_model = LayoutFID(
            max_num_elements=50,
            num_labels=9,
            net_path="/scipostlayout/code/layout-dm/download/fid_weights/FIDNetV3/scipostlayout-max50/model_best.pth.tar",
            device=device
        )
        max_iou_results = []
        alignment_results = []
        overlap_results = []

        with fs.open(pkl_path, "rb") as file_obj:
            x = pickle.load(file_obj)
        generated = x["results"]

        print("len of generated,", len(generated))
        for i in range(0, len(generated)):
            batch = generated[i]
            max_len = len(batch[-1])
            bbox, label, padding_mask, mask = preprocess([batch], max_len, device)
            batch_gold = layouts_main[i]
            max_len_gold = len(batch_gold[-1])
            bbox_gold, label_gold, padding_mask_gold, mask_gold = preprocess([batch_gold], max_len_gold, device)

            n_colors = 10
            colors = sns.color_palette('husl', n_colors=n_colors)
            draw_colors = [
                tuple(map(lambda x: int(x * 255), c)) for c in colors
            ]
            save_image(
                convert_ltwh_to_ltrb(bbox),
                label,
                mask,
                draw_colors,
                f"{args.result_dir}/{i}_gen.png",
                canvas_size=(360, 240)
            )

            overlap_eval = compute_overlap(convert_ltwh_to_ltrb(bbox).cpu(), mask.cpu())
            alignment_eval = compute_alignment(convert_ltwh_to_ltrb(bbox).cpu(), mask.cpu())
            max_iou_eval = compute_maximum_iou(
                label.squeeze(0).cpu(),
                convert_ltwh_to_ltrb(bbox).squeeze(0).cpu(),
                label_gold.squeeze(0).cpu(),
                convert_ltwh_to_ltrb(bbox_gold).squeeze(0).cpu(),
            )
            fid_model.collect_features(
                convert_ltwh_to_ltrb(bbox).to(device),
                label.to(device),
                padding_mask.to(device),
            )
            fid_model.collect_features(
                convert_ltwh_to_ltrb(bbox_gold).to(device),
                label_gold.to(device),
                padding_mask_gold.to(device),
                real=True
            )

            overlap_results.append(overlap_eval)
            alignment_results.append(alignment_eval)
            max_iou_results.append(max_iou_eval)

        print("aggregate results")
        print("overlap:", np.mean(overlap_results))
        print("alignment:", np.mean(alignment_results))
        print("max_iou:", np.mean(max_iou_results))
        try:
            print("FID: ", fid_model.compute_score_fid())
        except Exception as e:
            print("FID calcaulation failed")
        print("KID: ", fid_model.compute_score_kid())


if __name__ == "__main__":
    main()

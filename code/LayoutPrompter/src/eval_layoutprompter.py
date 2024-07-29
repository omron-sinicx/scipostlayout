import torch
import numpy as np
from tqdm import tqdm

from fid import LayoutFID
from utils import compute_overlap, compute_alignment, compute_maximum_iou


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fid_model = LayoutFID(
    max_num_elements=50,
    num_labels=9,
    net_path="/scipostlayout/code/layout-dm/download/fid_weights/FIDNetV3/scipostlayout-max50/model_best.pth.tar",
    device=device
)

result_path = "/scipostlayout/code/LayoutPrompter/results/gen_t"
gold_labels_list = torch.load(f"{result_path}/gold_labels.pth")
predictions_list = torch.load(f"{result_path}/predictions.pth")

max_iou_results = []
alignment_results = []
overlap_results = []

assert len(predictions_list) == 50, f"results broken, the length of predictions_list is {len(predictions_list)}"

for test_idx in tqdm(range(len(predictions_list))):
    fid_model.collect_features(
        predictions_list[test_idx]["_pred_bboxes"].to(device),
        predictions_list[test_idx]["_pred_labels"].to(device),
        ~predictions_list[test_idx]["_pred_padding_mask"].to(device),
    )
    fid_model.collect_features(
        gold_labels_list[test_idx]["_gold_bboxes"].to(device),
        gold_labels_list[test_idx]["_gold_labels"].to(device),
        ~gold_labels_list[test_idx]["_gold_padding_mask"].to(device),
        real=True
    )
    overlap_eval = compute_overlap(predictions_list[test_idx]["_pred_bboxes"], predictions_list[test_idx]["_pred_padding_mask"])
    alignment_eval = compute_alignment(predictions_list[test_idx]["_pred_bboxes"], predictions_list[test_idx]["_pred_padding_mask"])
    max_iou_eval = compute_maximum_iou(
        predictions_list[test_idx]["_pred_labels"].squeeze(0),
        predictions_list[test_idx]["_pred_bboxes"].squeeze(0),
        gold_labels_list[test_idx]["_gold_labels"].squeeze(0),
        gold_labels_list[test_idx]["_gold_bboxes"].squeeze(0),
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
except Exception:
    print("FID calcaulation failed")
print("KID: ", fid_model.compute_score_kid())

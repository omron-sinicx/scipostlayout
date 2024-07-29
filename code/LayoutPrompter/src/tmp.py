import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

from openai import OpenAI
from preprocess import create_processor
from fid import LayoutFID
from utils import RAW_DATA_PATH, read_pt, write_pt
from utils import convert_ltwh_to_ltrb, compute_overlap, compute_alignment, compute_maximum_iou
from selection import create_selector
from serialization import create_serializer, build_prompt
from parsing import Parser
from ranker import Ranker
from visualization import Visualizer, create_image_grid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gent", "gents", "genr", "completion", "refinement"])
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--fid_model_path", type=str)
    parser.add_argument("--use_prepared_constraints", action="store_true")
    parser.add_argument("--constraints_path", type=str)

    return parser.parse_args(sys.argv[1:])


args = parse_args()
print(f"RUNNING TASK: {args.task}")

# config
datasets = ["rico", "publaynet", "scipostlayout-max50"]  # choices

dataset = datasets[2]
input_format = "seq"
output_format = "html"
add_unk_token = False
add_index_token = True
add_sep_token = True
candidate_size = -1  # -1 represents the complete training set
num_prompt = 10

#  data preparation
processor = create_processor(dataset=dataset, task=args.task)


def get_processed_data(split):
    filename = os.path.join(
        args.base_dir, "datasets", dataset, "processed", args.task, f"{split}.pt"
    )
    if os.path.exists(filename):
        print("load processed data")
        processed_data = read_pt(filename)
    else:
        print("no processed data, perform processing")
        processed_data = []
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        raw_path = os.path.join(RAW_DATA_PATH(dataset), f"{split}.pt")
        raw_data = read_pt(raw_path)
        for rd in tqdm(raw_data, desc=f"{split} data processing..."):
            processed_data.append(processor(rd))
        write_pt(filename, processed_data)
    return processed_data


processed_train_data = get_processed_data("train")
processed_val_data = get_processed_data("val")
processed_test_data = get_processed_data("test")
print("preprocessing OK")

print(len(processed_test_data))
print(processed_val_data[0])


# evaluation preparation
gold_labels_list = []
predictions_list = []
for test_idx in range(len(processed_test_data)):
    gold_labels_list.append(
        {
            "labels": processed_test_data[test_idx]["labels"],
            "bboxes": processed_test_data[test_idx]["gold_bboxes"],
            "_gold_labels": processed_test_data[test_idx]["labels"].unsqueeze(0),
            "_gold_bboxes": convert_ltwh_to_ltrb(processed_test_data[test_idx]["gold_bboxes"]).unsqueeze(0),
            "_gold_padding_mask": torch.ones_like(processed_test_data[test_idx]["labels"].unsqueeze(0)).bool()
        }
    )
max_iou_results = []
alignment_results = []
overlap_results = []
fid_results = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
fid_model = LayoutFID(
    max_num_elements=50,
    num_labels=9,
    net_path=args.fid_model_path,
    device=device
)


# inference
for test_idx in tqdm(range(len(processed_test_data))):
    # dynamic exemplar selection
    selector = create_selector(
        task=args.task,
        train_data=processed_train_data,
        candidate_size=candidate_size,
        num_prompt=num_prompt,
    )
    exemplars = selector(processed_test_data[test_idx])
    print("examplar selection OK")

    # input-output serialization
    serializer = create_serializer(
        args=args,
        dataset=dataset,
        task=args.task,
        input_format=input_format,
        output_format=output_format,
        add_index_token=add_index_token,
        add_sep_token=add_sep_token,
        add_unk_token=add_unk_token
    )
    prompt = build_prompt(serializer, exemplars, processed_test_data[test_idx], dataset)
    print("serialization OK")

    # call gpt
    model = "gpt-3.5-turbo-1106"
    temperature = 0.7
    max_tokens = 4000
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0
    num_return = 10
    stop_token = "\n\n"
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORGANIZATION")
    )

    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                n=num_return,
                stop=[stop_token],
            )
            response = response.choices
            response = [r.message.content for r in response]
            print("GPT OK")

            # parsing
            parser = Parser(dataset=dataset, output_format=output_format)
            parsed_response = parser(response)
            print(f"filter {num_return - len(parsed_response)} invalid response", "parsing OK")

            # layout ranking
            val_path = os.path.join(RAW_DATA_PATH(dataset), "val.pt")
            ranker = Ranker(val_path=val_path)
            ranked_response = ranker(parsed_response)
            print("layout ranking OK")

            # evaluation
            top_response = ranked_response[0]
            _pred_labels = top_response[0].unsqueeze(0)
            _pred_bboxes = convert_ltwh_to_ltrb(top_response[1]).unsqueeze(0)
            _pred_padding_mask = torch.ones_like(_pred_labels).bool()
            predictions_list.append(
                {
                    "labels": top_response[0],
                    "bboxes": top_response[1],
                    "_pred_labels": _pred_labels,
                    "_pred_bboxes": _pred_bboxes,
                    "_pred_padding_mask": _pred_padding_mask
                }
            )
            overlap_eval = compute_overlap(_pred_bboxes, _pred_padding_mask)
            alignment_eval = compute_alignment(_pred_bboxes, _pred_padding_mask)
            max_iou_eval = compute_maximum_iou(
                top_response[0],
                top_response[1],
                [gold_labels_list[test_idx]["labels"]],
                [gold_labels_list[test_idx]["bboxes"]],
            )
            fid_model.collect_features(
                _pred_bboxes.to(device),
                _pred_labels.to(device),
                ~_pred_padding_mask.to(device)
            )
            fid_model.collect_features(
                gold_labels_list[test_idx]["_gold_bboxes"].to(device),
                gold_labels_list[test_idx]["_gold_labels"].to(device),
                ~gold_labels_list[test_idx]["_gold_padding_mask"].to(device),
                real=True
            )
            overlap_results.append(overlap_eval)
            alignment_results.append(alignment_eval)
            max_iou_results.append(max_iou_eval)
            print("evaluatoin ok")
            break
        except Exception:
            print("all sample failed, generate again")

    # visualization
    visualizer = Visualizer(dataset)
    images = visualizer(ranked_response)
    image_grid = create_image_grid(images)
    Path(f"grid/{args.task}").mkdir(parents=True, exist_ok=True)
    image_grid.save(f"grid/{args.task}/test_{test_idx}.png")
    print("visualization OK")


# evaluation
print("aggregate results")
print("overlap:", np.mean(overlap_results))
print("alignment:", np.mean(alignment_results))
print("max_iou:", np.mean(max_iou_results))
try:
    print("FID: ", fid_model.compute_score_fid())
except Exception:
    print("FID calcaulation failed")
print("KID: ", fid_model.compute_score_kid())


# save results
Path(f"results/{args.task}").mkdir(parents=True, exist_ok=True)
torch.save(gold_labels_list, f"results/{args.task}/gold_labels.pth")
torch.save(predictions_list, f"results/{args.task}/predictions.pth")
print("results saved")

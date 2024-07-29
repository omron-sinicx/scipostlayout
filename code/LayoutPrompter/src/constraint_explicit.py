import os
import sys
import torch
import numpy as np
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import argparse
import json
from copy import deepcopy
import glob

from openai import OpenAI
from preprocess import create_processor
from fid import LayoutFID
from utils import RAW_DATA_PATH, ID2LABEL, read_pt, write_pt
from utils import convert_ltwh_to_ltrb, compute_overlap, compute_alignment, compute_maximum_iou
from selection import create_selector
from serialization import create_serializer, build_prompt
from parsing import Parser
from ranker import Ranker
from visualization import Visualizer, create_image_grid, save_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gent", "gents", "genr", "completion", "refinement", "genp"])
    parser.add_argument("--base_dir", type=str)
    parser.add_argument("--fid_model_path", type=str)
    parser.add_argument("--gen_const_path", type=str)
    parser.add_argument("--mmd_dir", type=str)
    parser.add_argument("--use_saved_response", action="store_true")

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

n_colors = 10
colors = sns.color_palette('husl', n_colors=n_colors)
draw_colors = [
    tuple(map(lambda x: int(x * 255), c)) for c in colors
]


def gpt_summarize(paper_text):
    model = "gpt-4-1106-preview"
    temperature = 0.7
    max_tokens = 4000
    top_p = 1
    frequency_penalty = 0
    presence_penalty = 0
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORGANIZATION")
    )
    prompt = f"""
Please summarize the following paper within 1000 words.
The summary does not need to include all elements of the paper, but should prioritize important elements such as proposed methods and main experimental results.
The summary should include the title and the author names of the paper.
Use the same wording as in the paper's abstract.
DO NOT generate redundant messages.
"""
    retry_count = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": paper_text}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
            )
            response = response.choices[0].message.content
            if len(response.split(" ")) < 100:
                raise RuntimeError
            break
        except Exception:
            retry_count += 1
            print("Retry!")
        if retry_count > 5:
            response = paper_text
            break
    return response


if args.task == "genp":
    mmd_dir = args.mmd_dir
    assert mmd_dir != ""
    split_list = ["dev", "test"]
    split_paper_summary_dic = {}

    for split in split_list:
        mmd_paths = glob.glob(f"{mmd_dir}/{split}/*.mmd")
        summary_path = f"{mmd_dir}/{split}/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                paper_summary_dic = json.loads(f.read())

            # retry_list = ["10434", "6202", "14300"]
            # retry_list = ["10434"]
            # for key, summary in paper_summary_dic.items():
            #     if key not in retry_list:
            #         continue
            #     for mmd_path in mmd_paths:
            #         name = os.path.basename(mmd_path).split(".")[0]
            #         if key == name:
            #             with open(mmd_path) as f:
            #                 paper = f.read()
            #             break
            #     assert key == name
            #     summary = gpt_summarize(paper)
            #     paper_summary_dic[name] = summary

            # with open(summary_path, "w") as f:
            #     json.dump(paper_summary_dic, f, indent=4)
        else:
            paper_summary_dic = {}
            print("Summarizing papers...")
            for mmd_path in tqdm(mmd_paths):
                with open(mmd_path) as f:
                    paper = f.read()
                name = os.path.basename(mmd_path).split(".")[0]
                summary = gpt_summarize(paper)
                paper_summary_dic[name] = summary

            with open(summary_path, "w") as f:
                json.dump(paper_summary_dic, f, indent=4)

        split_paper_summary_dic[split] = paper_summary_dic

    split_paper_summary_dic["val"] = split_paper_summary_dic["dev"]


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
            if args.task == "genp" and split != "train":
                try:
                    rd["text"] = split_paper_summary_dic[split][rd["name"].split(".")[0]]
                except KeyError:
                    continue
            processed_data.append(processor(rd))
        write_pt(filename, processed_data)
    return processed_data


processed_train_data = get_processed_data("train")
processed_val_data = get_processed_data("val")
processed_test_data = get_processed_data("test")
print("preprocessing OK")


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

Path(f"results/{args.task}").mkdir(parents=True, exist_ok=True)

index2label = ID2LABEL[dataset]
label2index = {v: k for k, v in index2label.items()}

gen_const_path = args.gen_const_path
if gen_const_path is not None:
    with open(gen_const_path, "r") as f:
        gen_const_list = json.loads(f.read())

    new_test_data = []
    for data in processed_test_data:
        for const_data in gen_const_list:
            if const_data["id"] == data["name"]:
                break
        if const_data["id"] != data["name"]:
            continue
        labels = []
        for category, num in const_data["result_dict"].items():
            num = int(num)
            labels.extend([label2index[category] for _ in range(num)])
        new_data = deepcopy(data)
        new_data["labels"] = torch.tensor(labels)
        new_test_data.append(new_data)

    processed_test_data = new_test_data


use_saved_response = args.use_saved_response
saved_response_path = f"results/{args.task}/saved_response.jsonl"
if use_saved_response and os.path.exists(saved_response_path):
    with open(saved_response_path, "r") as f:
        response_list = [json.loads(l.strip()) for l in f.readlines()]
else:
    response_list = []


# inference
for test_idx in tqdm(range(len(processed_test_data))):
    if args.task == "genp":
        example_data = processed_val_data
    else:
        example_data = processed_train_data
    # dynamic exemplar selection
    selector = create_selector(
        task=args.task,
        train_data=example_data,
        candidate_size=candidate_size,
        num_prompt=num_prompt,
    )
    exemplars = selector(processed_test_data[test_idx])
    print("examplar selection OK")

    # input-output serialization
    serializer = create_serializer(
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
    # model = "gpt-3.5-turbo-1106"
    model = "gpt-4-1106-preview"
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

    saved_response = None
    if use_saved_response:
        for response in response_list:
            if response["name"] == processed_test_data[test_idx]["name"]:
                saved_response = response
                break

    break_count = 0
    while True:
        try:
            if saved_response is not None:
                response = saved_response["response"]
                print("Use saved response")
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    n=num_return,
                    # stop=[stop_token],
                )
                response = response.choices
                response = [r.message.content for r in response]
                print("GPT OK")

            # print("prompt:\n", prompt[0]["content"])
            # for r in response:
            #     print("response:\n", r)

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
                convert_ltwh_to_ltrb(top_response[1]),
                gold_labels_list[test_idx]["labels"],
                convert_ltwh_to_ltrb(gold_labels_list[test_idx]["bboxes"]),
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

            save_image(
                _pred_bboxes,
                _pred_labels,
                _pred_padding_mask,
                draw_colors,
                f"results/{args.task}/{test_idx}_gen.png",
                canvas_size=(360, 240)
            )

            if saved_response is None:
                response_list.append({
                    "name": processed_test_data[test_idx]["name"],
                    "response": response
                })

                with open(saved_response_path, "w") as f:
                    for res in response_list:
                        f.write(json.dumps(res))
                        f.write("\n")

            break
        except Exception:
            break_count += 1
            if break_count < 3:
                print("all sample failed, generate again")
            else:
                print("failed many times, skip this example.")
                break

    # visualization grid
    if break_count < 3:
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

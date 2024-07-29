from openai import OpenAI
import os
import sys
import re
import json
from tqdm import tqdm
from glob import glob
import argparse


def gpt_call(client, prompt, paper, model="gpt-4-1106-preview"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": paper}
            ]
        )
    except Exception:
        print("OPENAI is down.")
        sys.exit(0)
    response = completion.choices[0].message.content
    return response


def cal_mae(results):
    category_results = {
            "Title": [],
            "Author Info": [],
            "Section": [],
            "Text": [],
            "List": [],
            "Table": [],
            "Figure": [],
            "Caption": []
        }
    all_results = []
    for data in results:
        for key in data["label_dict"].keys():
            category_results[key].append(abs(data["label_dict"][key] - data["result_dict"][key]))
            all_results.append(abs(data["label_dict"][key] - data["result_dict"][key]))
    for key in category_results.keys():
        print(f"MAE of {key}: {sum(category_results[key]) / len(category_results[key])}")
    print(f"MAE of ALL: {sum(all_results) / len(all_results)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--mmd_path", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")

    return parser.parse_args(sys.argv[1:])


args = parse_args()
mmd_paths = glob(args.mmd_path + "/*.mmd")
with open(args.prompt_path) as f:
    prompt = f.read()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    organization=os.environ.get("OPENAI_ORGANIZATION")
)

with open(args.data_path) as f:
    data = json.load(f)
category_dict = {}
for cate in data["categories"]:
    category_dict[cate["id"]] = cate["name"]
label_dicts = {}
image_id_dict = {}
for image in data["images"]:
    image_id = image["file_name"].split(".")[0]
    image_id_dict[image["id"]] = image_id
    label_dicts[image_id] = {
        "Title": 0,
        "Author Info": 0,
        "Section": 0,
        "Text": 0,
        "List": 0,
        "Table": 0,
        "Figure": 0,
        "Caption": 0
    }
for anno in data["annotations"]:
    if anno["category_id"] == 3591166397:
        continue  # Unknown-pos
    image_id = image_id_dict[anno["image_id"]]
    label_dicts[image_id][category_dict[anno["category_id"]]] += 1

results = []
for mmd_path in tqdm(mmd_paths):
    idx = mmd_path.split('/')[-1].split('.')[0]
    label_dict = label_dicts[idx]
    result_dict = label_dict.copy()
    for key in result_dict:
        result_dict[key] = 0
    gpt_response = []

    with open(mmd_path) as f:
        paper = f.read()

    ok_n = 0
    while True:
        try:
            response = gpt_call(client, prompt, paper, args.model)
            pattern = r'```json(.*?)```'
            match = re.findall(pattern, response, re.DOTALL)[0][1:]
            match_dict = json.loads(match)
            for key, value in match_dict.items():
                result_dict[key] += value
            gpt_response.append(response)
            ok_n += 1
            if ok_n == 3:
                break
        except Exception:
            print("GPT4 output error.")
    for key in result_dict.keys():
        result_dict[key] /= 3.0

    results.append(
        {
            "id": idx,
            "label_dict": label_dict,
            "result_dict": result_dict,
            "GPT_response": gpt_response
        }
    )

split = "dev" if "dev" in args.mmd_path else "test"
file_name = f"results/{split}/{args.prompt_path.split('/')[-1].split('.')[0]}.json"
os.makedirs(os.path.dirname(file_name), exist_ok=True)
with open(file_name, "w") as f:
    json.dump(results, f, indent=4)

# with open(file_name, "r") as f:
#     results = json.loads(f.read())

cal_mae(results)

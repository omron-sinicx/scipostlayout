import os.path as osp
import json
import shutil

data_dir = "../../scipostlayout/"

with open(osp.join(data_dir, "nouhin0129.json")) as f:
    data = json.load(f)

with open(osp.join(data_dir, "train.json")) as f:
    old_train_data = json.load(f)

with open(osp.join(data_dir, "dev.json")) as f:
    old_dev_data = json.load(f)

with open(osp.join(data_dir, "test.json")) as f:
    old_test_data = json.load(f)

train = {}
dev = {}
test = {}

train["categories"] = dev["categories"] = test["categories"] = data["categories"]
train["images"] = []
dev["images"] = []
test["images"] = []
train["annotations"] = []
dev["annotations"] = []
test["annotations"] = []

split_dict = {}

for idx, img_data in enumerate(old_train_data["images"]):
    filename = img_data["file_name"]
    if idx < 450:
        split_dict[filename] = "dev"
        shutil.move(osp.join(data_dir, "train", filename), osp.join(data_dir, "dev", filename))
    elif idx < 900:
        split_dict[filename] = "test"
        shutil.move(osp.join(data_dir, "train", filename), osp.join(data_dir, "test", filename))
    else:
        split_dict[filename] = "train"

for img_data in old_dev_data["images"]:
    split_dict[img_data["file_name"]] = "dev"

for img_data in old_test_data["images"]:
    split_dict[img_data["file_name"]] = "test"

id_dict = {}
for img in data["images"]:
    if split_dict[img["file_name"]] == "train":
        train["images"].append(img)
        id_dict[img["id"]] = "train"
    elif split_dict[img["file_name"]] == "dev":
        dev["images"].append(img)
        id_dict[img["id"]] = "dev"
    elif split_dict[img["file_name"]] == "test":
        test["images"].append(img)
        id_dict[img["id"]] = "test"

for anno in data["annotations"]:
    if id_dict[anno["image_id"]] == "train":
        train["annotations"].append(anno)
    elif id_dict[anno["image_id"]] == "dev":
        dev["annotations"].append(anno)
    elif id_dict[anno["image_id"]] == "test":
        test["annotations"].append(anno)

with open(osp.join(data_dir, "train.json"), "w") as f:
    json.dump(train, f, indent=4)
with open(osp.join(data_dir, "dev.json"), "w") as f:
    json.dump(dev, f, indent=4)
with open(osp.join(data_dir, "test.json"), "w") as f:
    json.dump(test, f, indent=4)

with open(osp.join(data_dir, "train.txt"), "w") as f:
    for img in data["images"]:
        if split_dict[img["file_name"]] == "train":
            f.write(img["file_name"])
            f.write("\n")

with open(osp.join(data_dir, "dev.txt"), "w") as f:
    for img in data["images"]:
        if split_dict[img["file_name"]] == "dev":
            f.write(img["file_name"])
            f.write("\n")

with open(osp.join(data_dir, "test.txt"), "w") as f:
    for img in data["images"]:
        if split_dict[img["file_name"]] == "test":
            f.write(img["file_name"])
            f.write("\n")

import json

with open("nouhin20231228.json") as f:
    data = json.load(f)

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
with open("nouhin20231228_list.csv") as f:
    img_list = f.readlines()
    img_list = [img.strip() for img in img_list]
    for img in img_list:
        split, img_name = img.split(",")[0], img.split(",")[1]
        split_dict[img_name] = split

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

with open("train_20231228.json", "w") as f:
    json.dump(train, f)
with open("dev_20231228.json", "w") as f:
    json.dump(dev, f)
with open("test_20231228.json", "w") as f:
    json.dump(test, f)

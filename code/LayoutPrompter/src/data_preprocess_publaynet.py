from pathlib import Path
from pycocotools.coco import COCO

import torch
from torch_geometric.data import Data

from data.base import BaseDataset


class PubLayNet(BaseDataset):
    labels = [
        'text',
        'title',
        'list',
        'table',
        'figure',
    ]

    def __init__(self, split='train', transform=None):
        super().__init__('publaynet', split, transform)

    def download(self):
        super().download()

    def process(self):
        raw_dir = "/scratch/acd13848fc/paper2poster/SciPostLayout/poster_png"
        for split_publaynet in ['train', 'dev', 'test']:
            data_list = []
            coco = COCO(raw_dir / f'{split_publaynet}.json')
            for img_id in sorted(coco.getImgIds()):
                ann_img = coco.loadImgs(img_id)
                W = float(ann_img[0]['width'])
                H = float(ann_img[0]['height'])
                name = ann_img[0]['file_name']
                # if H < W:
                #     continue

                def is_valid(element):
                    x1, y1, width, height = element['bbox']
                    x2, y2 = x1 + width, y1 + height
                    if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                        return False

                    if x2 <= x1 or y2 <= y1:
                        return False

                    return True

                elements = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
                _elements = list(filter(is_valid, elements))
                filtered = len(elements) != len(_elements)
                elements = _elements

                N = len(elements)
                if N == 0 or 9 < N:
                    continue

                boxes = []
                labels = []

                for element in elements:
                    # bbox
                    x1, y1, width, height = element['bbox']
                    xc = x1 + width / 2.
                    yc = y1 + height / 2.
                    b = [xc / W, yc / H,
                         width / W, height / H]
                    boxes.append(b)

                    # label
                    l = coco.cats[element['category_id']]['name']
                    labels.append(self.label2index[l])

                boxes = torch.tensor(boxes, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)

                data = Data(x=boxes, y=labels)
                data.attr = {
                    'name': name,
                    'width': W,
                    'height': H,
                    'filtered': filtered,
                    'has_canvas_element': False,
                }
                data_list.append(data)

            if split_publaynet == 'train':
                train_list = data_list
            else:
                val_list = data_list

        # shuffle train with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(train_list), generator=generator)
        train_list = [train_list[i] for i in indices]

        # train_list -> train 95% / val 5%
        # val_list -> test 100%
        s = int(len(train_list) * .95)
        torch.save(self.collate(train_list[:s]), self.processed_paths[0])
        torch.save(self.collate(train_list[s:]), self.processed_paths[1])
        torch.save(self.collate(val_list), self.processed_paths[2])


PubLayNet.process()
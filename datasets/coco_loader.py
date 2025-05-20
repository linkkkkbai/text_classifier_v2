import os
import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import numpy as np
import torchvision.transforms as T


class CropBBoxDataset(Dataset):
    def __init__(self, cfg, json_file, is_train=True, input_size=224):
        self.cfg = cfg
        self.root_dir = cfg.data.root_dir
        self.is_train = is_train
        self.input_size = input_size

        # 加载COCO格式数据
        with open(os.path.join(self.root_dir, json_file), 'r') as f:
            self.coco_data = json.load(f)

        # 属性到索引的映射
        self.attr_maps = {
            '类型': {cls: i for i, cls in enumerate(cfg.attributes.type.classes)},
            '字体': {cls: i for i, cls in enumerate(cfg.attributes.font.classes)},
            '斜体': {cls: i for i, cls in enumerate(cfg.attributes.italic.classes)}
        }

        # 构建所有bbox的样本列表
        self.samples = []
        imgid_to_imginfo = {img['id']: img for img in self.coco_data['images']}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            img_info = imgid_to_imginfo[img_id]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            bbox = ann['bbox']  # [x, y, w, h]
            label = [
                self.attr_maps['类型'][ann['attributes']['类型']],
                self.attr_maps['字体'][ann['attributes']['字体']],
                self.attr_maps['斜体'][ann['attributes']['斜体']]
            ]
            self.samples.append((img_path, bbox, label))

        if self.is_train:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.Resize((self.input_size, self.input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((self.input_size, self.input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, bbox, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x, y, w, h = [int(round(v)) for v in bbox]
        H, W = image.shape[:2]
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = max(1, min(w, W-x))
        h = max(1, min(h, H-y))
        if w <= 0 or h <= 0:
            crop = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        else:
            crop = image[y:y+h, x:x+w, :]
            crop = cv2.resize(crop, (self.input_size, self.input_size))
        crop = self.transform(crop)
        label = torch.tensor(label, dtype=torch.long)
        return crop, label
    
def build_dataloader(cfg, json_file, batch_size=32, is_train=True, num_workers=4, input_size=224, shuffle=None):
    dataset = CropBBoxDataset(cfg, json_file, is_train=is_train, input_size=input_size)
    if shuffle is None:
        shuffle = is_train
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
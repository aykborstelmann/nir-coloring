import numpy as np
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from nircoloring.dataset.caltech import AbstractMetaDataSource


def parse_label_map(meta_data_source: AbstractMetaDataSource, entries: pd.Series):
    images = meta_data_source.load_images()
    annotations = meta_data_source.load_annotations()

    images["base_file_name"] = images["file_name"].str.lower().str.rpartition("/")[2]
    images = images[images["base_file_name"].isin(entries.str.lower())]
    labels = annotations.groupby("image_id")["category_id"].unique()
    images = images.merge(labels, how="left", left_on="id", right_on="image_id")
    return images[["base_file_name", "category_id"]].set_index("base_file_name")


class SerengetiDataset(Dataset):
    def __init__(self, path, meta_data_source, transform, size) -> None:
        self.path = path
        self.entries: pd.Series = pd.Series(os.listdir(path))
        self.label_map: pd.DataFrame = parse_label_map(meta_data_source, self.entries)
        self.transform = transform
        self.size = size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index) -> T_co:
        filename = self.entries[index]
        img_file_path = os.path.join(self.path, filename)
        img = Image.open(img_file_path).convert('RGB')
        img = self.transform(img)

        labels_indices = self.label_map["category_id"][filename.lower()]

        return {
            'image': img,
            'labels': torch.tensor(labels_indices[0], dtype=torch.float32)
        }

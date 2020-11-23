import torch
import os
import numpy as np
from PIL import Image
import pandas as pd


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, pics_dir, transform=None, csv_header=None):
        self.annotations = pd.read_csv(csv_path, sep=" ", header=csv_header)
        self.root_dir = pics_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.annotations.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, filename)
        image = Image.open(img_name).convert("RGB")
        labels = self.annotations.iloc[idx, 1:]
        labels = np.array(labels)
        labels = torch.tensor(labels.astype("float"))

        if self.transform:
            image = self.transform(image)

        return filename, image, labels

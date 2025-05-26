import os
import cv2
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import timm
from src.utils import get_logger


class myModel(nn.Module):
    def __init__(self,
                 arch_name,
                 pretrained=False,
                 img_size=256,
                 multi_drop=False,
                 multi_drop_rate=0.5,
                 att_layer=False,
                 att_pattern="A"
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.att_layer = att_layer
        self.multi_drop = multi_drop
        self.arch_name = arch_name  # 儲存架構名稱以便在 forward 中使用

        # 創建模型
        self.model = timm.create_model(
            arch_name, pretrained=pretrained, img_size=img_size
        )
        
        # 根據架構名稱動態獲取特徵維度
        if 'vit' in arch_name:
            n_features = self.model.head.in_features  # ViT: 768
            self.model.head = nn.Identity()  # 移除 ViT 的頭部
        elif 'swin' in arch_name:
            n_features = self.model.head.in_features  # Swin: 1024
            self.model.head = nn.Identity()  # 移除 Swin 的頭部
        else:
            raise ValueError(f"Unsupported architecture: {arch_name}")

        # 定義分類頭部
        self.head = nn.Linear(n_features, 5)
        self.head_drops = nn.ModuleList()
        for i in range(5):
            self.head_drops.append(nn.Dropout(multi_drop_rate))

        # 定義注意力層
        if att_layer:
            if att_pattern == "A":
                self.att_layer = nn.Sequential(
                    nn.Linear(n_features, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1),
                )
            elif att_pattern == "B":
                self.att_layer = nn.Linear(n_features, 1)
            else:
                raise ValueError("invalid att pattern")

    def forward(self, x):
        if self.att_layer:
            l = x.shape[2] // 2
            # 分割輸入圖像為四個部分
            h1 = self.model(x[:, :, :l, :l])
            h2 = self.model(x[:, :, :l, l:])
            h3 = self.model(x[:, :, l:, :l])
            h4 = self.model(x[:, :, l:, l:])

            # 對 Swin Transformer 的特徵圖進行全局平均池化
            if 'swin' in self.arch_name:
                h1 = h1.mean(dim=(1, 2))  # (batch_size, 12, 12, 1024) -> (batch_size, 1024)
                h2 = h2.mean(dim=(1, 2))
                h3 = h3.mean(dim=(1, 2))
                h4 = h4.mean(dim=(1, 2))

            # 計算注意力權重
            w = F.softmax(torch.cat([
                self.att_layer(h1),
                self.att_layer(h2),
                self.att_layer(h3),
                self.att_layer(h4),
            ], dim=1), dim=1)
            h = h1 * w[:, 0].unsqueeze(-1) + \
                h2 * w[:, 1].unsqueeze(-1) + \
                h3 * w[:, 2].unsqueeze(-1) + \
                h4 * w[:, 3].unsqueeze(-1)
        else:
            h = self.model(x)
            # 對 Swin Transformer 的特徵圖進行全局平均池化
            if 'swin' in self.arch_name:
                h = h.mean(dim=(1, 2))  # (batch_size, 12, 12, 1024) -> (batch_size, 1024)

        # 多重 Dropout 頭部
        if self.multi_drop:
            for i, dropout in enumerate(self.head_drops):
                if i == 0:
                    output = self.head(dropout(h))
                else:
                    output += self.head(dropout(h))
            output /= len(self.head_drops)
        else:
            output = self.head(h)
        return output


class myDataset(Dataset):
    def __init__(self,
        settings,
        df,
        transform=None,
    ):
        self.settings = settings
        self.img_ids = df["image_id"].values
        self.labels = df["label"].values
        self.transform = transform
        self.logger = get_logger()
        self.logger.info(f"Dataset initialized with {len(self.img_ids)} samples")
        self.logger.info(f"Sample labels: {self.labels[:5]}")
        self.logger.info(f"Label dtype: {self.labels.dtype}")

    def __len__(self):
        return len(self.img_ids)

    def load_img(self, path):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=image)
        return augmented['image']

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # Ensure label is a scalar integer
        label = np.array(self.labels[idx]).item()
        label = torch.tensor(label, dtype=torch.long)
        self.logger.debug(f"Index {idx}: Label value = {label}, Label shape = {label.shape}, Label type = {label.dtype}")
        path = f"{self.settings.DATA_PATH}/train/{img_id}"
        return self.load_img(path), label


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CrossEntropyLossWithLabelSmooth(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

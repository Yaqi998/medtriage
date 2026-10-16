"""Shared model, transform and data helpers for the pneumonia chest X-ray classifier."""
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


class PneumoniaNet(nn.Module):
    """ImageNet ResNet18; stem+layer1+layer2 stay frozen, layer3+layer4+head are fine-tuned."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.frozen = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2)
        self.trainable = nn.Sequential(
            base.layer3, base.layer4, base.avgpool, nn.Flatten(1), nn.Dropout(0.2), nn.Linear(512, 1)
        )
        for p in self.frozen.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # frozen BatchNorm layers must keep their ImageNet running statistics
        self.frozen.eval()
        return self

    def forward(self, x):
        return self.trainable(self.frozen(x)).squeeze(1)


def eval_transform(img_size: int):
    return T.Compose([
        T.Resize(round(img_size * 256 / 224)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def train_transform(img_size: int):
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomRotation(7),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def read_manifest(path, split=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


class XrayDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tf):
        self.rows = rows
        self.tf = tf

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r["proc_path"]) as im:
            x = self.tf(im.convert("RGB"))
        return x, float(r["label"])


def load_checkpoint(model_dir, name="pneumonia_resnet18"):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / f"{name}.json").read_text())
    model = PneumoniaNet(pretrained=False)
    model.load_state_dict(torch.load(model_dir / f"{name}.pt", map_location="cpu"))
    model.eval()
    return model, meta

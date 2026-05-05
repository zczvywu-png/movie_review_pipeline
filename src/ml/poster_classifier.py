"""
海报类型分类（迁移学习）
================================

设计依据
- 老师 WS5 用 Keras MobileNet 微调（最后 3 层 trainable，4 epoch，flow_from_directory）。
- 本项目改用 PyTorch + torchvision 的 MobileNetV3-Small：
    * 避免 Windows 上 TensorFlow 的安装坑；
    * 与图像向量化模块共用 PyTorch 环境；
    * MobileNetV3-Small 在 CPU 上比 MobileNetV1 还快、参数更少。
- 训练逻辑完全对应老师代码：冻结 backbone → 微调最后分类头 → 4-5 epoch。
- 数据来源：data/posters/{genre}/*.jpg，目录结构同老师 WS5 的 `images/`。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms
from tqdm import tqdm

from src.utils import get_logger, get_path, load_config

log = get_logger("poster_clf")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PosterDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []
        for c in self.classes:
            for p in (self.root / c).glob("*.jpg"):
                self.samples.append((p, self.class_to_idx[c]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def _build_model(n_classes: int) -> nn.Module:
    """老师 WS5 的"冻结 backbone 只微调分类头"对应到 PyTorch 写法"""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    for p in model.parameters():
        p.requires_grad = False
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, n_classes)
    return model.to(DEVICE)


def train_poster_classifier() -> dict:
    cfg = load_config()
    posters_root = get_path(cfg["output"]["posters_dir"])
    if not any(posters_root.iterdir()):
        raise RuntimeError(f"{posters_root} 为空，请先跑 tmdb_scraper")

    img_size = cfg["vectorize"]["image"]["image_size"]
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    full_train = PosterDataset(posters_root, transform=train_tf)
    full_eval = PosterDataset(posters_root, transform=eval_tf)
    n_total = len(full_train)
    n_val = max(1, n_total // 5)
    n_tr = n_total - n_val
    indices = list(range(n_total))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    tr_idx, val_idx = indices[:n_tr], indices[n_tr:]

    log.info(f"数据集：{n_total} 张，{len(full_train.classes)} 类，train/val={n_tr}/{n_val}")

    from torch.utils.data import Subset

    train_loader = DataLoader(
        Subset(full_train, tr_idx),
        batch_size=cfg["ml"]["poster_batch_size"],
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(full_eval, val_idx),
        batch_size=cfg["ml"]["poster_batch_size"],
        shuffle=False,
        num_workers=0,
    )

    model = _build_model(len(full_train.classes))
    optim = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    crit = nn.CrossEntropyLoss()

    epochs = cfg["ml"]["poster_train_epochs"]
    history = {"train_loss": [], "val_acc": []}

    for ep in range(epochs):
        model.train()
        losses = []
        for x, y in tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optim.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        # 验证
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        train_loss = float(np.mean(losses))
        history["train_loss"].append(train_loss)
        history["val_acc"].append(acc)
        log.info(f"epoch {ep+1}: loss={train_loss:.4f}  val_acc={acc:.4f}")

    save_dir = get_path("outputs/models/poster_clf")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": full_train.classes,
            "history": history,
        },
        save_dir / "mobilenet_v3_small.pt",
    )
    with open(save_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump({"classes": full_train.classes, "history": history}, f, indent=2)

    log.info(f"模型已保存 → {save_dir}")
    return {"classes": full_train.classes, "history": history}


if __name__ == "__main__":
    train_poster_classifier()

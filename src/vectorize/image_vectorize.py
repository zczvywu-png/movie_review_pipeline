"""
图像向量化模块（CLIP vs ResNet50 对比）
================================

设计依据
- 作业明确要求"至少两种向量化技术对比（例如 CLIP vs ResNet50）"，本模块直接对应这一条。
- CLIP 调用方式沿用老师 WS8 的写法：
    `CLIPModel.from_pretrained("openai/clip-vit-base-patch32")`
- ResNet50 用 torchvision 自带的 ImageNet 预训练权重，去掉最后 fc 层取 2048-d 特征。
- 对比指标和文本模块一致：聚类纯度 + KNN mAP@10，保证对比可量化、可比较。

为什么选 CLIP vs ResNet50
- ResNet50：纯视觉特征，靠 ImageNet 监督训练，关注"图像里有什么物体"。
- CLIP：图文对比学习，特征空间和语言对齐，关注"这张图在描述什么场景/风格"。
- 海报恰好是"风格 + 物体"混合体，两者会得到截然不同的聚类结构，
  能给报告写出"基于自己的结果而非理论"的对比段落。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from tqdm import tqdm

from src.utils import get_logger, get_path, load_config

log = get_logger("img_vec")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"使用设备：{DEVICE}")


# ---------- ResNet50 提取器 ----------
class ResNet50Extractor:
    """torchvision ResNet50（ImageNet 预训练）→ 2048-d 全局平均池化特征"""

    def __init__(self):
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.model = models.resnet50(weights=weights).to(DEVICE).eval()
        # 去掉最后 fc 层，取 avgpool 输出
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.transform = weights.transforms()
        self.dim = 2048

    @torch.no_grad()
    def encode(self, image_paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="ResNet50"):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    imgs.append(self.transform(img))
                except Exception as e:
                    log.warning(f"读图失败 {p}: {e}")
                    imgs.append(torch.zeros(3, 224, 224))
            batch = torch.stack(imgs).to(DEVICE)
            feat = self.feature_extractor(batch).squeeze(-1).squeeze(-1)
            feats.append(feat.cpu().numpy().astype(np.float32))
        return np.vstack(feats)


# ---------- CLIP 提取器（沿用老师 WS8 写法）----------
class CLIPExtractor:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(model_name).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = self.model.config.projection_dim  # 512 for base-patch32

    @torch.no_grad()
    def encode(self, image_paths: list[str | Path], batch_size: int = 16) -> np.ndarray:
        feats = []
        for i in tqdm(range(0, len(image_paths), batch_size), desc="CLIP"):
            batch_paths = image_paths[i : i + batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception as e:
                    log.warning(f"读图失败 {p}: {e}")
                    imgs.append(Image.new("RGB", (224, 224)))
            inputs = self.processor(images=imgs, return_tensors="pt").to(DEVICE)
            feat = self.model.get_image_features(**inputs)
            # 归一化（CLIP 推荐用法）
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            feats.append(feat.cpu().numpy().astype(np.float32))
        return np.vstack(feats)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """额外暴露文本编码，做"评论 ↔ 海报"跨模态检索时用"""
        all_feats = []
        for i in range(0, len(texts), 32):
            batch = texts[i : i + 32]
            inputs = self.processor(
                text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77
            ).to(DEVICE)
            feat = self.model.get_text_features(**inputs)
            feat = feat / feat.norm(p=2, dim=-1, keepdim=True)
            all_feats.append(feat.cpu().numpy().astype(np.float32))
        return np.vstack(all_feats)


# ---------- 对比 ----------
def compare_image_vectorizers(
    images_csv: str | Path | None = None,
    save_dir: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    在同一批海报上跑 CLIP 和 ResNet50，输出对比指标。
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import LabelEncoder

    from src.vectorize.text_vectorize import cluster_purity, knn_retrieval_map

    images_csv = Path(images_csv) if images_csv else get_path("data/raw/tmdb_images.csv")
    save_dir = Path(save_dir) if save_dir else get_path("outputs/models/image_vec")
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(images_csv)
    # 转回绝对路径（CSV 里存的是相对项目根的路径）
    df["abs_path"] = df["local_path"].apply(lambda p: str(get_path(p)))
    df = df[df["abs_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
    log.info(f"图像数据集：{len(df)} 张，覆盖 {df['movie_title'].nunique()} 部电影")

    paths = df["abs_path"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df["genre"].astype(str).values)
    n_clusters = len(le.classes_)

    extractors = {
        "clip": CLIPExtractor(),
        "resnet50": ResNet50Extractor(),
    }

    matrices: dict[str, np.ndarray] = {}
    rows = []
    for name, ext in extractors.items():
        log.info(f"--- 提取 {name} 特征 ---")
        mat = ext.encode(paths)
        matrices[name] = mat
        np.save(save_dir / f"{name}_matrix.npy", mat)

        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = km.fit_predict(mat)
        purity = cluster_purity(y, cluster_labels)
        map_at_10 = knn_retrieval_map(mat, y, k=10)
        rows.append(
            {
                "vectorizer": name,
                "dim": mat.shape[1],
                "cluster_purity": round(purity, 4),
                "knn_mAP@10": round(map_at_10, 4),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("knn_mAP@10", ascending=False)
    metrics.to_csv(save_dir / "comparison.csv", index=False)
    df.to_csv(save_dir / "image_index.csv", index=False, encoding="utf-8-sig")
    log.info("\n" + metrics.to_string(index=False))
    return matrices, metrics


if __name__ == "__main__":
    compare_image_vectorizers()

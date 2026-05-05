"""
通用工具函数：配置加载、路径管理、日志、文本清洗
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_logger(name: str = "pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)s %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_path(relative: str) -> Path:
    """相对项目根的路径转绝对路径，并确保父目录存在"""
    p = PROJECT_ROOT / relative
    if p.suffix:
        ensure_dir(p.parent)
    else:
        ensure_dir(p)
    return p


# ---------- 文本清洗 ----------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<[^>]+>")
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002500-\U00002BEF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(s: str, keep_emoji: bool = False) -> str:
    """轻量清洗：去 URL / HTML / 多余空白；可选保留 emoji"""
    if not isinstance(s, str):
        return ""
    s = _URL_RE.sub(" ", s)
    s = _HTML_RE.sub(" ", s)
    if not keep_emoji:
        s = _EMOJI_RE.sub(" ", s)
    s = _MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def is_meaningful(text: str, min_words: int = 3, min_chars: int = 10) -> bool:
    """过滤掉过短、纯符号、垃圾评论"""
    if not text:
        return False
    if len(text) < min_chars:
        return False
    words = text.split()
    if len(words) < min_words:
        return False
    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.5:
        return False
    return True


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)

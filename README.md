# Movie Review Analysis Pipeline

> 一条端到端的电影评论分析工作流：抓取 → 向量化 → GPT 增强 → 机器学习 → 可视化。
> 在老师的 5 个 Workshop notebook（WS1 Selenium / WS4 Vectorising / WS5 Image Classification / WS5 Video / WS8 GPT）基础上重构整合而来，主题聚焦"YouTube 电影预告片评论 + 多模态分析"。

## 一、对应作业要求

| 作业要求 | 本项目实现 |
|----------|------------|
| 网页抓取：≥2 个网站/域，3+ 数据集，每个 200-300 元素 | YouTube 评论（12 部电影 × 250 条 ≈ 3000 条）+ IMDb 影评（12 × 60 ≈ 720 条）+ TMDB 海报（12 × 12 ≈ 144 张） |
| 向量化：至少两种方法对比 | **文本**：TF-IDF vs Word2Vec vs Doc2Vec；**图像**：CLIP vs ResNet50 |
| 基于自己的结果而非理论 | 每组对比都跑了**聚类纯度** + **KNN mAP@10** 两个量化指标 |
| API 交互（创建自己的脚本） | 真实调用 OpenAI Chat Completions API，做情感打标 + 合成评论 + ReAct Agent |
| 机器学习 | K-Means 聚类 / LogisticRegression+LinearSVC+RandomForest 情感分类 / MobileNetV3 海报迁移学习 |
| 可视化 | 7 张图：评论量分布 / 情感堆叠柱+小提琴 / 词云 / 文本 t-SNE / 海报 t-SNE / 海报网格 / 跨模态热力图 |
| 工作流闭环 | `python main.py all` 一键串跑；`demo.ipynb` 分步演示 |

## 二、目录结构

```
movie_review_pipeline/
├── configs/
│   └── config.yaml          # 电影列表、抓取规模、模型超参
├── .env.example             # API Key 模板（改名为 .env 后填入）
├── requirements.txt
├── main.py                  # CLI 入口
├── notebooks/
│   └── demo.ipynb           # Notebook 演示版
├── src/
│   ├── utils.py             # 配置/路径/清洗
│   ├── scrapers/
│   │   ├── youtube_scraper.py
│   │   ├── imdb_scraper.py
│   │   ├── tmdb_scraper.py
│   │   └── merge_data.py
│   ├── vectorize/
│   │   ├── text_vectorize.py    # TF-IDF / W2V / Doc2Vec + 对比
│   │   └── image_vectorize.py   # CLIP / ResNet50 + 对比
│   ├── api/
│   │   ├── gpt_client.py        # OpenAI 封装 + 重试
│   │   ├── sentiment_analysis.py
│   │   ├── synthetic_reviews.py
│   │   └── movie_agent.py       # ReAct Agent（4 个工具）
│   ├── ml/
│   │   ├── text_clustering.py
│   │   ├── sentiment_classifier.py
│   │   └── poster_classifier.py
│   └── viz/
│       └── visualize.py
├── data/
│   ├── raw/                 # 抓取原始 CSV
│   ├── processed/           # 合并 / 打标 / 增强后的 CSV
│   └── posters/{genre}/     # TMDB 海报，按类型分目录（与老师 WS5 images/ 同结构）
└── outputs/
    ├── figures/             # 可视化 PNG
    └── models/              # 向量矩阵 + 训练好的模型
```

## 三、快速开始

### 1) 安装依赖

```bash
cd movie_review_pipeline
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### 2) 配置 API Key

```bash
copy .env.example .env          # Windows
# 编辑 .env，填入 OPENAI_API_KEY 和 TMDB_API_KEY
```

- TMDB Key：免费注册 https://www.themoviedb.org/settings/api
- OpenAI Key：https://platform.openai.com/api-keys ；国内访问可用 `OPENAI_BASE_URL` 切到中转服务

### 3) 一键运行

```bash
# 冒烟测试（每视频抓 5 条，5 分钟内验证管线通畅）
python main.py smoke

# 全量运行（约 60-90 分钟）
python main.py all
```

### 4) 单步运行

```bash
python main.py scrape       # 抓取
python main.py merge        # 合并
python main.py vectorize    # 向量化对比
python main.py annotate     # GPT 情感打标
python main.py augment      # GPT 合成评论
python main.py ml           # 聚类 + 分类
python main.py viz          # 可视化
python main.py agent "Recommend a movie similar to John Wick"
```

## 四、与老师 Workshop 代码的对应关系

| 老师 Workshop | 本项目对应模块 | 复用情况 |
|----------|--------|----------|
| WS1 Selenium | `src/scrapers/youtube_scraper.py` + `imdb_scraper.py` | 沿用 BeautifulSoup 模式 + 限速礼仪；YouTube 改用 `youtube-comment-downloader` 命中 InnerTube API（比 Selenium 抓 Shadow DOM 稳得多） |
| WS4 Vectorising | `src/vectorize/text_vectorize.py` | TF-IDF / Word2Vec / Doc2Vec 三段几乎照搬，把 epub 输入换成评论字符串 |
| WS5 Image Classification | `src/ml/poster_classifier.py` | 同样的"冻结 backbone + 微调最后一层"思路；把 Keras MobileNet 换成 PyTorch MobileNetV3-Small（避免 Win+Py3.11 装 TF 的坑） |
| WS5 Video & SAM | **未启用** | 你的项目主题是评论文本，2.4 GB SAM 权重 + 视频特征不相关，可省去 AutoDL 开销 |
| WS8 GPT | `src/api/sentiment_analysis.py` + `synthetic_reviews.py` + `movie_agent.py` + `image_vectorize.py(CLIP)` | CLIP 调用沿用老师 `CLIPModel.from_pretrained(...)`；**真实接入 OpenAI SDK**（老师只贴了 ChatGPT 网页输出，本项目实现 API + Agent，符合作业"创建您自己的脚本"的要求） |

## 五、电影类型选择

`configs/config.yaml` 中按 3 个类型各选 4 部，覆盖近年高热度电影预告片：

- **action**: Mission: Impossible 7, John Wick 4, Top Gun: Maverick, Furiosa
- **romance**: La La Land, Past Lives, Anyone But You, The Notebook
- **horror**: Hereditary, Get Out, Smile, Talk to Me

可在 yaml 中替换为任意 YouTube 视频 ID + IMDb ID + TMDB ID。

## 六、运行时间与算力评估

| 步骤 | 本地 CPU 用时 | 是否需要 GPU |
|------|:---:|:---:|
| 抓取（YouTube + IMDb + TMDB） | 30-45 分钟 | 否 |
| 文本向量化对比 | 3-8 分钟 | 否 |
| 图像向量化对比（CLIP+ResNet50） | 5-10 分钟 | 否 |
| GPT 打标 300 条 | 3-5 分钟（取决于 OpenAI 响应） | 否 |
| 合成 240 条评论 | 2-4 分钟 | 否 |
| 聚类 + 情感分类 | 1-2 分钟 | 否 |
| 海报分类（5 epoch） | 8-15 分钟 | 否（GPU 1 分钟） |
| 可视化 | 2-5 分钟 | 否 |
| **合计** | **60-90 分钟** |  |

## 七、潜在风险与降级方案

| 风险 | 降级方案 |
|------|----------|
| YouTube 评论抓取被限流 | 降低 `comments_per_video`，加大 `request_delay_sec` |
| IMDb 改版导致选择器失效 | `imdb_scraper.py` 已尝试 3 种 fallback 选择器；最坏情况只用 YouTube + TMDB 评分作为第二数据源 |
| OpenAI 国内访问失败 | 在 `.env` 设置 `OPENAI_BASE_URL` 指向中转 |
| 海报数据不足以训练 | `poster_classifier.py` 的 train/val 自动按 80/20 切分；增大 TMDB `max_posters` 至 12-15 |

## 八、引用与致谢

- 老师提供的 5 个 Workshop notebook 提供了核心算法骨架。
- 第三方开源工具：`youtube-comment-downloader`, `transformers`, `torchvision`, `gensim`, `sklearn`, `openai`。

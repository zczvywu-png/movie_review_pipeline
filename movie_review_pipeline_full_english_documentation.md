# Movie Review Analysis Pipeline

> An end-to-end movie review analysis workflow: scraping → vectorization → GPT enhancement → machine learning → visualization.
>
> This project is refactored and integrated from the instructor’s five workshop notebooks (WS1 Selenium / WS4 Vectorising / WS5 Image Classification / WS5 Video / WS8 GPT), with a primary focus on **YouTube movie trailer comments and multimodal analysis**.

---

# 1. Project Overview

This project builds a complete data analysis and machine learning pipeline around movie reviews and movie-related media content.

The workflow includes:

1. Web scraping from multiple platforms
2. Text and image vectorization
3. GPT-based sentiment annotation and data augmentation
4. Machine learning for clustering and classification
5. Multimodal visualization and analysis

The project is designed to satisfy academic coursework requirements while also demonstrating a practical end-to-end AI workflow.

---

# 2. Assignment Requirement Mapping

| Assignment Requirement | Project Implementation |
|----------|------------|
| Web scraping: at least 2 websites/domains, 3+ datasets, each containing 200–300 elements | YouTube comments (12 movies × 250 comments ≈ 3000 comments) + IMDb reviews (12 × 60 ≈ 720 reviews) + TMDB posters (12 × 12 ≈ 144 posters) |
| Compare at least two vectorization methods | **Text**: TF-IDF vs Word2Vec vs Doc2Vec; **Images**: CLIP vs ResNet50 |
| Use experimental results instead of pure theory | Each comparison includes **cluster purity** and **KNN mAP@10** quantitative evaluation metrics |
| API interaction with self-written scripts | Real OpenAI Chat Completions API integration for sentiment labeling, synthetic review generation, and a ReAct Agent |
| Machine learning implementation | K-Means clustering / LogisticRegression + LinearSVC + RandomForest sentiment classification / MobileNetV3 poster transfer learning |
| Data visualization | 7 visual outputs: comment distribution, sentiment stacked bar chart, violin plot, word cloud, text t-SNE, poster t-SNE, poster grid, and cross-modal heatmap |
| End-to-end workflow | `python main.py all` provides one-click execution; `demo.ipynb` provides step-by-step notebook demonstration |

---

# 3. Dataset Sources

The project uses three primary data sources:

## 3.1 YouTube Trailer Comments

- Source: YouTube movie trailer videos
- Data collected:
  - Comment text
  - Like count
  - Publish date
  - User information
- Scale:
  - 12 movies
  - Approximately 250 comments per movie
  - Around 3000 total comments

## 3.2 IMDb Reviews

- Source: IMDb movie review pages
- Data collected:
  - User reviews
  - Ratings
  - Review titles
- Scale:
  - Approximately 60 reviews per movie
  - Around 720 total reviews

## 3.3 TMDB Posters

- Source: TMDB API
- Data collected:
  - Movie posters
  - Genre metadata
  - Movie information
- Scale:
  - Approximately 12 posters per movie
  - Around 144 total poster images

---

# 4. Project Structure

```text
movie_review_pipeline/
├── configs/
│   └── config.yaml          # Movie list, scraping scale, model hyperparameters
├── .env.example             # API key template (rename to .env and fill in)
├── requirements.txt
├── main.py                  # CLI entry point
├── notebooks/
│   └── demo.ipynb           # Notebook demonstration version
├── src/
│   ├── utils.py             # Config/path handling/cleaning
│   ├── scrapers/
│   │   ├── youtube_scraper.py
│   │   ├── imdb_scraper.py
│   │   ├── tmdb_scraper.py
│   │   └── merge_data.py
│   ├── vectorize/
│   │   ├── text_vectorize.py
│   │   └── image_vectorize.py
│   ├── api/
│   │   ├── gpt_client.py
│   │   ├── sentiment_analysis.py
│   │   ├── synthetic_reviews.py
│   │   └── movie_agent.py
│   ├── ml/
│   │   ├── text_clustering.py
│   │   ├── sentiment_classifier.py
│   │   └── poster_classifier.py
│   └── viz/
│       └── visualize.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── posters/{genre}/
└── outputs/
    ├── figures/
    └── models/
```

---

# 5. Environment Setup

## 5.1 Create Virtual Environment

```bash
cd movie_review_pipeline
python -m venv .venv
```

## 5.2 Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

## 5.3 Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 6. API Configuration

## 6.1 Create Environment File

```bash
copy .env.example .env
```

## 6.2 Configure Keys

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_key
TMDB_API_KEY=your_tmdb_key
OPENAI_BASE_URL=optional_proxy_url
```

## 6.3 API References

### TMDB API

https://www.themoviedb.org/settings/api

### OpenAI API

https://platform.openai.com/api-keys

---

# 7. Running the Project

## 7.1 Smoke Test

Quick pipeline validation:

```bash
python main.py smoke
```

Features:

- Downloads only a few comments
- Runs within approximately 5 minutes
- Useful for debugging the pipeline

## 7.2 Full Pipeline

```bash
python main.py all
```

Expected runtime:

- Approximately 60–90 minutes

---

# 8. Individual Pipeline Commands

## 8.1 Scraping

```bash
python main.py scrape
```

Functions:

- Scrape YouTube comments
- Scrape IMDb reviews
- Download TMDB posters

---

## 8.2 Merge Data

```bash
python main.py merge
```

Functions:

- Merge datasets
- Normalize columns
- Clean missing values

---

## 8.3 Text and Image Vectorization

```bash
python main.py vectorize
```

Functions:

- TF-IDF encoding
- Word2Vec encoding
- Doc2Vec encoding
- CLIP embedding
- ResNet50 feature extraction

---

## 8.4 GPT Sentiment Annotation

```bash
python main.py annotate
```

Functions:

- GPT-based sentiment labeling
- Positive/neutral/negative classification
- OpenAI API integration

---

## 8.5 Synthetic Review Generation

```bash
python main.py augment
```

Functions:

- Generate synthetic reviews
- Expand training dataset
- Improve class balance

---

## 8.6 Machine Learning

```bash
python main.py ml
```

Functions:

- K-Means clustering
- Sentiment classification
- Poster classification

---

## 8.7 Visualization

```bash
python main.py viz
```

Functions:

- Generate all visual outputs
- Create plots and embedding visualizations

---

## 8.8 ReAct Movie Agent

```bash
python main.py agent "Recommend a movie similar to John Wick"
```

Functions:

- GPT-powered reasoning agent
- Movie recommendation
- Multi-tool workflow execution

---

# 9. Text Vectorization Methods

## 9.1 TF-IDF

TF-IDF converts text into sparse vectors based on word frequency and inverse document frequency.

Advantages:

- Fast
- Interpretable
- Strong baseline performance

Disadvantages:

- Cannot capture semantic relationships effectively

---

## 9.2 Word2Vec

Word2Vec learns dense semantic word embeddings.

Advantages:

- Captures semantic similarity
- Efficient dense representation

Disadvantages:

- Requires larger datasets
- Word-level only

---

## 9.3 Doc2Vec

Doc2Vec generates document-level embeddings.

Advantages:

- Better sentence/document representation
- Captures contextual information

Disadvantages:

- Longer training time
- More hyperparameter sensitivity

---

# 10. Image Vectorization Methods

## 10.1 CLIP

CLIP provides multimodal text-image embeddings.

Advantages:

- Strong semantic alignment
- Excellent zero-shot performance
- Supports cross-modal retrieval

---

## 10.2 ResNet50

ResNet50 extracts deep CNN image features.

Advantages:

- Stable image representations
- Strong classification performance

Disadvantages:

- No direct text-image alignment

---

# 11. Machine Learning Models

## 11.1 K-Means Clustering

Used for:

- Discovering latent review groups
- Evaluating embedding quality

Evaluation metrics:

- Cluster purity
- t-SNE visualization

---

## 11.2 Sentiment Classification

Implemented models:

- Logistic Regression
- LinearSVC
- RandomForest

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score

---

## 11.3 Poster Classification

Model:

- MobileNetV3-Small

Training strategy:

- Freeze backbone
- Fine-tune classification head

Advantages:

- Lightweight
- Fast training
- CPU-friendly

---

# 12. Visualization Outputs

The project generates seven major visualization outputs.

## 12.1 Comment Distribution Chart

Shows the number of comments collected for each movie.

---

## 12.2 Sentiment Stacked Bar Chart

Displays sentiment proportions across movies.

---

## 12.3 Violin Plot

Visualizes sentiment score distributions.

---

## 12.4 Word Cloud

Highlights high-frequency review keywords.

---

## 12.5 Text t-SNE Embedding

Projects text embeddings into 2D space.

---

## 12.6 Poster t-SNE Embedding

Projects poster image embeddings into 2D space.

---

## 12.7 Cross-Modal Heatmap

Displays relationships between text and image embeddings.

---

# 13. Workshop Mapping

| Workshop | Corresponding Module | Notes |
|----------|--------|----------|
| WS1 Selenium | `youtube_scraper.py` + `imdb_scraper.py` | Reuses BeautifulSoup structure and polite crawling strategy |
| WS4 Vectorising | `text_vectorize.py` | Reuses TF-IDF / Word2Vec / Doc2Vec workflow |
| WS5 Image Classification | `poster_classifier.py` | Transfer learning using MobileNetV3 |
| WS5 Video & SAM | Not used | Video analysis omitted because it is unrelated to review-focused workflow |
| WS8 GPT | GPT modules + CLIP modules | Real OpenAI SDK integration and agent implementation |

---

# 14. Selected Movie Genres

## Action

- Mission: Impossible 7
- John Wick 4
- Top Gun: Maverick
- Furiosa

## Romance

- La La Land
- Past Lives
- Anyone But You
- The Notebook

## Horror

- Hereditary
- Get Out
- Smile
- Talk to Me

These can be modified in `configs/config.yaml`.

---

# 15. Runtime Estimation

| Step | CPU Runtime | GPU Required |
|------|:---:|:---:|
| Scraping | 30–45 min | No |
| Text vectorization | 3–8 min | No |
| Image vectorization | 5–10 min | No |
| GPT annotation | 3–5 min | No |
| Synthetic review generation | 2–4 min | No |
| Clustering and classification | 1–2 min | No |
| Poster classification | 8–15 min | Optional |
| Visualization | 2–5 min | No |
| Total | 60–90 min | Optional |

---

# 16. Risks and Fallback Strategies

| Risk | Fallback Strategy |
|------|----------|
| YouTube rate limiting | Reduce comment count and increase request delay |
| IMDb page structure changes | Use fallback selectors |
| OpenAI unavailable locally | Configure proxy endpoint using `OPENAI_BASE_URL` |
| Insufficient poster data | Increase `max_posters` in config |

---

# 17. Technical Highlights

This project demonstrates:

- Multimodal AI workflow design
- Real-world API integration
- End-to-end machine learning engineering
- Data scraping and preprocessing
- NLP and computer vision integration
- GPT-powered automation
- ReAct Agent implementation
- Transfer learning
- Visualization and quantitative evaluation

---

# 18. Future Improvements

Potential future improvements include:

- Real-time streaming review analysis
- Transformer-based sentiment classification
- Multilingual review processing
- Video trailer feature extraction
- Recommendation system integration
- RAG-based movie knowledge assistant
- Fine-tuned domain-specific LLMs

---

# 19. References and Acknowledgements

## Instructor Materials

The instructor’s five workshop notebooks provided the foundational algorithmic structure for this project.

## Open-Source Libraries

Main third-party libraries used:

- youtube-comment-downloader
- transformers
- torchvision
- gensim
- sklearn
- openai
- pandas
- numpy
- matplotlib
- seaborn
- torch

---

# 20. Conclusion

This project successfully builds a complete multimodal movie review analysis pipeline that integrates:

- Web scraping
- NLP vectorization
- Computer vision
- GPT enhancement
- Machine learning
- Data visualization

The system demonstrates both academic and practical value by combining modern AI workflows with real-world movie review data.

It also satisfies all major coursework requirements while extending the original workshop materials into a significantly more complete engineering project.


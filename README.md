# Multimodal RAG for Movie Search & Recommendation

> Retrieval-Augmented Generation System with Text + Image Understanding

Authors: [**Mykhailo Ivasiuk**](https://github.com/Fenix125), [**Maksym Dzoban**](https://github.com/MaxDzioban)

## Overview

In this project we implemented a Multimodal Retrieval-Augmented Generation (RAG) system for movie discovery.  
Users can search for movies using:

-   Text descriptions (plot, mood, genre, atmosphere)
-   Poster style descriptions (colors, layout, visual mood)
-   Actual poster images (image-to-image similarity search)
-   Combined multimodal queries (text + poster style)

The system retrieves relevant movies from a ChromaDB vector database and uses an LLM agent to produce final recommendations.

## Motivation

Traditional movie recommenders rely only on text metadata, but movie posters also contain rich semantic information:

-   Mood, color palette
-   Genre cues
-   Visual style, composition

Users often think visually (e.g., _“a dark neon cyberpunk vibe”_, _“a pastel cozy romance poster”_).

Goals of the project:

-   Support text-based search
-   Support poster-style search (image or textual description)
-   Enable unified multimodal retrieval

## System Architecture

The system consists of three main parts:

### 1. Dataset Ingestion & Preprocessing

-   Load movie dataset (metadata + posters)
-   Filter incomplete/bad records
-   Save cleaned JSONL + poster images

### 2. Vector Indexing (ChromaDB)

Two vector collections are used:

| Collection              | Contents                                | Purpose                  |
| ----------------------- | --------------------------------------- | ------------------------ |
| `movies_overviews_text` | chunks of movies description + metadata | text semantic search     |
| `movies_posters_images` | images + poster metadata                | poster similarity search |

### 3. LLM Agent

The agent uses two custom tools:

-   **movie_multimodal_search** — unified text + image search
-   **image_search** — poster-based similarity

The tools merge results, deduplicates entries and produces the final RAG answer.

---

### Text Embeddings

We use models such as E5-base for superior retrieval quality.

Supported models:

-   `sentence-transformers/all-mpnet-base-v2`
-   `intfloat/multilingual-e5-base`
-   `intfloat/multilingual-e5-large`

### Image Embeddings (CLIP)

Vision models:

-   `openai/clip-vit-base-patch32`
-   `openai/clip-vit-base-patch16`

## Retrieval Tools

### **Movie Multimodal Search Tool**

Accepts:

-   `text_query` (plot/mood/genre description)
-   `image_query` (poster style text)

Performs:

-   Text search
-   Poster-style search
-   Merging
-   Deduplication
-   Output

### **Image Search Tool**

Accepts:

-   `image_path` (local file)

Processes:

-   Image embedding creation
-   Vector Store Search

---

## LLM Agent

Built using **LangChain**:

-   Configurable provider (OpenAI / Gemini)
-   Custom system prompt
-   Produces RAG-based final answers

---

### Systems Requirements

-   Python: 3.12

Create and activate a virtual environment

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Environment Variables:

#optional

-   `GEMINI_MODEL`=gemini-2.5-flash
-   `OPENAI_MODEL`=gpt-4o-mini
-   `TEXT_EMBED_MODEL`=intfloat/multilingual-e5-base
-   `CLIP_MODEL`=openai/clip-vit-base-patch32
-   `CHROMA_PATH`=data/chroma
-   `STREAMLIT_PORT`=9000

#necessary (one of, if both provided uses google api by default)

-   `GOOGLE_AI_API_KEY`=your_api_key_here
-   `OPENAI_API_KEY`=your_api_key_here

> [!IMPORTANT]
> Make sure at least GEMINI_API_KEY is set - without it the agent won't work

> [!NOTE]
> All articles and the Chroma db are stored locally under `CHROMA_PATH`

## Run:

Option A - Let the UI handle ingestion

Start Streamlit:

```shell
streamlit run app.py
```

-   Open the app in your browser (usually http://localhost:8501).
-   Go to the Settings tab in the sidebar.
-   Click “Rebuild movie index” then “Yes, rebuild”.

This will:

-   Download the dataset
-   Build text + image embeddings.
-   Store everything in Chroma under `CHROMA_PATH`

Switch back to Chat and start asking questions!

Option B - Ingest via CLI scripts:

```shell
# 1. Fetch raw data into data/processed/movies.jsonl
python -m src.scripts.preprocess_movies

# 2. Build text + image embeddings and write to Chroma
python -m src.scripts.ingest_chroma_db

# 3. Run the Streamlit app
streamlit run app.py
```

Optional: Docker Deployment

Run with Docker:

```shell
docker compose up --build
```

Useful commands:

```shell
docker compose stop      # stop containers
docker compose start     # start them again
docker compose down      # stop + remove containers
```

> [!NOTE]
> The app-data volume keeps your Chroma DB between restarts,
> so you don’t repeatedly re-download and re-embed everything.


## Evaluation (Optional)

If you want to reproduce the evaluation:

1. src/scripts/evaluate_movies_rag.py:
   Runs the agent over a dataset in data_test/metrics
   Computes Recall@K, Precision@K, MRR, Hit@K
   Writes \*\_with_results.json and prints a small table
2. src/scripts/evaluate_rag_deepeval.py
   Works on dataset in data_test/deepeval
   Uses DeepEval to compute:
    - ContextualRecall
    - ContextualRelevancy
    - AnswerRelevancy
    - Faithfulness


> [!NOTE]
> You will need to setup `OPENAI_API_KEY` to use DeepEval evaluation

```shell
python -m src.scripts.evaluate_movies_rag --help
python -m src.scripts.evaluate_rag_deepeval --help
```

(See each script’s --help for exact arguments.)

## License

This project is open-sourced.
See the LICENSE file in the repository for full details.

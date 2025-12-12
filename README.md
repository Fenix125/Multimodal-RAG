# Multimodal-RAG

start chat: python3 -m agent.chat

## Dataset

load dataset: python3 -m agent.preprocess_data


# Multimodal RAG for Movie Search & Recommendation  
## Retrieval-Augmented Generation System with Text + Image Understanding  
### Team 2 — Authors: **Mykhailo Ivasiuk**, **Maksym Dzoban**

## Overview
In this NLP course project we implemented a Multimodal Retrieval-Augmented Generation (RAG) system for movie discovery.  
Users can search for movies using:

- Text descriptions (plot, mood, genre, atmosphere)  
- Poster style descriptions (colors, layout, visual mood)  
- Actual poster images (image-to-image similarity search)  
- Combined multimodal queries (text + poster style)

The system retrieves relevant movies from a vector database ChromaDB and uses an LLM agent to produce final recommendations.

## Motivation
Traditional movie recommenders rely only on text metadata, but movie posters also contain rich semantic information:

- Mood, color palette  
- Genre cues  
- Visual style, composition  

Users often think visually (e.g., _“a dark neon cyberpunk vibe”_, _“a pastel cozy romance poster”_).  

Goals of the project:

- Support text-based search  
- Support poster-style search (image or textual description)  
- Enable unified multimodal retrieval  

## System Architecture
The system consists of three main parts:

#### 1. Dataset Ingestion & Preprocessing
- Load movie dataset (metadata + posters)  
- Filter incomplete/bad records  
- Chunk overview text  
- Generate text embeddings (E5 / MPNet)  
- Generate image embeddings (CLIP)  
- Save cleaned JSONL + poster images  

#### 2. Vector Indexing (ChromaDB)
Two vector collections are used:

| Collection                | Contents                                 | Purpose                     |
|--------------------------|--------------------------------------------|-----------------------------|
| `movies_overviews_text` | chunks of movie overviews + metadata       | text semantic search        |
| `movies_posters_images` | CLIP image embeddings + poster metadata    | poster similarity search    |

### 3. LLM Agent (LangChain)
The agent uses two custom tools:

- **movie_multimodal_search** — unified text + image search  
- **image_search** — poster-based similarity  

The agent merges results, deduplicates entries, enriches metadata, and produces the final RAG answer.


---

## Text Embeddings
We use models such as E5-base for superior retrieval quality.

Supported models:

- `sentence-transformers/all-mpnet-base-v2`
- `intfloat/multilingual-e5-base`
- `intfloat/multilingual-e5-large`

**Pipeline:**
1. Accept string or list  
2. Convert to list  
3. For E5 models:  
   - Queries → `"query: ..."`  
   - Documents → `"passage: ..."`  
4. Encode with SentenceTransformers  
5. L2-normalize  
6. Return embeddings for ChromaDB  

---

## Image Embeddings (CLIP)
Vision models:

- `openai/clip-vit-base-patch32`
- `openai/clip-vit-base-patch16`

**Pipeline:**
1. Load image (path/URL → PIL RGB)  
2. Resize, crop, normalize  
3. Encode via CLIP Vision Transformer  
4. Extract embedding  
5. L2-normalize  
6. Store in vector DB  

---

## Retrieval Tools

### **Movie Multimodal Search Tool**
Accepts:  
- `text_query` (plot/mood/genre description)  
- `image_query` (poster style text)  

Performs:  
- Text search (E5)  
- Poster-style search (CLIP text encoder)  
- Merging & ranking  
- Deduplication  
- JSON output  

### **Image Search Tool**
Accepts:  
- `image_path` (local uploaded poster)

Processes:  
- Load → preprocess  
- CLIP embedding extraction  
- KNN search in poster collection  
- Return movie metadata  

---

## LLM Agent
Built using **LangChain**:

- Configurable provider (OpenAI / Gemini)  
- Custom system prompt  
- Produces RAG-based final answers  

---

## 🐳 Running with Docker
```bash
docker-compose up --build
```

Or manually:
```
docker build -t multimodal-rag .
docker run -p 8000:8000 multimodal-rag
```

---

### Systems Requirements

- Python: 3.11 (required)
- OS: MacOS/ Linux

Create and activate a virtual environment

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install deepeval
```

Create a .env file (or export variables manually):

```
OPENAI_API_KEY=your_openai_api_key_here

```

## Run:

Preprocess dataset

```
python3 -m src.scripts.preprocess_movies
```

Output:

```
data/processed/movies.jsonl
```

Ingest data into ChromaDB:

```
python3 -m src.scripts.ingest_chroma_db
```

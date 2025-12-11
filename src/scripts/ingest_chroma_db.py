from __future__ import annotations

from pathlib import Path

from src.config import config
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.movies.ingestor import load_movies_from_jsonl
from src.vector_db.movie_chroma_store import MovieChromaIndexer

MOVIES_JSONL_PATH = Path("data/processed/movies.jsonl")

def main():
    print(f"[STEP] Loading movies from {MOVIES_JSONL_PATH}")
    movies = load_movies_from_jsonl(MOVIES_JSONL_PATH)
    print(f"[INFO] Loaded {len(movies)} movies")

    text_embedder = TextEmbedder(
        model_name=config.text_embed_model_name,
        device=config.device,
    )
    image_embedder = ImageEmbedder(
        model_name=config.clip_model_name,
        device=config.device,
    )

    indexer = MovieChromaIndexer(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )

    print("[STEP] Indexing movies into Chroma...")
    indexer.index(movies)
    print("[DONE] Indexing complete.")


if __name__ == "__main__":
    main()

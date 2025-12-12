from __future__ import annotations

from pathlib import Path

from src.movies.ingestor import ingest_movies_to_jsonl

MOVIES_JSONL_PATH = Path("data/processed/movies.jsonl")

def main():
    print("[STEP] Ingesting movies and saving JSONL + posters...")
    ingest_movies_to_jsonl(
        output_path=MOVIES_JSONL_PATH,
        sample_size=3000,
        min_overview_len=20,
    )
    print("[DONE] Preprocessing complete.")

if __name__ == "__main__":
    main()

from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Iterable, List

from datasets import load_dataset
from PIL import Image

from src.movies.templates import Movie

DS_NAME = "Pablinho/movies-dataset"

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def valid_rows(stream: Iterable[dict], min_len: int = 20) -> Iterable[dict]:
    """
    Filter out movies with very short or empty overviews.
    """
    for row in stream:
        overview = (row.get("Overview") or "").strip()
        if len(overview) >= min_len:
            yield row


def parse_genres(genre_raw: str) -> List[str]:
    if not genre_raw or not isinstance(genre_raw, str):
        return []
    return [g.strip() for g in genre_raw.split(",") if g.strip()]


def build_movie_from_row(idx: int, row: dict) -> Movie:
    """
    Build a Movie object from a raw dataset row.
    Uses idx as a stable movie_id (cast to string).
    """
    movie_id = str(idx)

    title = (row.get("Title") or "").strip() or "(untitled)"
    overview = (row.get("Overview") or "").strip()

    genres = parse_genres(row.get("Genre"))

    poster_url = (row.get("Poster_Url") or "").strip() or None

    release_date = (row.get("Release_Date") or "").strip() or None

    if not (movie_id and title and overview and genres and poster_url and release_date):
        return False, None

    return True, Movie(
        movie_id=movie_id,
        title=title,
        overview=overview,
        genres=genres,
        poster_url=poster_url,
        release_date=release_date
    )


def ingest_movies_to_jsonl(output_path: Path, sample_size: int = 3000, min_overview_len: int = 20) -> List[Movie]:
    """
    Load a subset of the movie_posters_100k dataset, filter by overview length,
    save posters to disk, and write a JSONL file with Movie records.

    JSONL schema per line (Movie.model_dump_json()):
      {
        "movie_id": "...",
        "title": "...",
        "genres": ["..."],
        "overview": "...",
        "poster_url": "https://...",
      }
    """
    print(f"[INFO] Loading dataset: {DS_NAME}")
    stream = load_dataset(DS_NAME, split="train", streaming=True)

    filtered = valid_rows(stream, min_len=min_overview_len)
    sample = list(islice(filtered, sample_size))

    print(f"[INFO] Building Movie objects for {len(sample)} rows")
    movies: List[Movie] = []
    for idx, row in enumerate(sample):
        success, movie = build_movie_from_row(idx, row)
        if success:
            movies.append(movie)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for movie in movies:
            f.write(movie.model_dump_json() + "\n")

    print(f"[DONE] Saved {len(movies)} valid movies to {output_path}")
    return movies


def load_movies_from_jsonl(path: Path) -> List[Movie]:
    """
    Load Movie records from a JSONL file written by ingest_movies_to_jsonl.
    """
    movies: List[Movie] = []
    if not path.exists():
        raise FileNotFoundError(f"Movies JSONL not found at {path}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            movies.append(Movie.model_validate_json(line))
    return movies

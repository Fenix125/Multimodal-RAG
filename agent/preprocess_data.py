import torch
import numpy as np
from itertools import islice
from datasets import load_dataset, Dataset
from agent.vector_store import (
    NAMESPACE_POSTER,
    NAMESPACE_TEXT,
    ensure_index,
    batch_upsert,
)
from agent.model import clip, text_embedding, img_embedding
from pathlib import Path

DS_NAME = "stzhao/movie_posters_100k_controlnet"

POSTERS_DIR = Path("data/posters")
POSTERS_DIR.mkdir(parents=True, exist_ok=True)

def get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device_name()

def run_preprocess_and_upsert():
    def valid_rows(rows, min_len=20):
        for r in rows:
            descr = (r.get("overview") or "").strip()
            if len(descr) >= min_len:
                yield r

    sample_size = 2000
    batch_size = 64
    
    print(f"Loading dataset ({sample_size} samples): {DS_NAME}")
    stream = load_dataset(DS_NAME, split="train", streaming=True)

    filtered_stream = valid_rows(stream, min_len=20)
    
    sample = list(islice(filtered_stream, sample_size))
    
    movie_ds = Dataset.from_list(sample)
    movie_ds = movie_ds.select_columns(["id", "image", "title", "genres", "overview"])

    ids = [str(row["id"]) for row in movie_ds]
    texts = [row["overview"] for row in movie_ds]
    posters = [row["image"] for row in movie_ds]
    
    poster_paths = []
    for i, pid in enumerate(ids):
        out_path = POSTERS_DIR / f"{pid}.jpg"
        if not out_path.exists():
            posters[i].save(out_path, format="JPEG", quality=100)
        poster_paths.append(str(out_path))

    print("Loading CLIP model")
    model, processor = clip()
    
    print("Computing text embeddings")
    text_vecs = text_embedding(model, processor, texts, batch_size)

    print("Computing image embeddings")
    img_vecs = img_embedding(model, processor, posters, batch_size)

    text_metadata = []
    img_metadata = []
    for i in range(len(ids)):
        text_metadata.append(
            {
                "id": ids[i],
                "title": movie_ds[i]["title"],
                "genres": [genre["name"] for genre in movie_ds[i]["genres"]],
                "overview": movie_ds[i]["overview"],
            }
        )
        img_metadata.append(
            {
                "id": ids[i],
                "poster_path": poster_paths[i],
            }
        )
    
    text_payload = []
    img_payload = []

    for i in range(len(ids)):
        text_payload.append((ids[i], text_vecs[i].tolist(), text_metadata[i]))
        img_payload.append((ids[i], img_vecs[i].tolist(), img_metadata[i]))

    batch_upsert(text_payload, namespace=NAMESPACE_TEXT, batch_size=batch_size)
    batch_upsert(img_payload, namespace=NAMESPACE_POSTER, batch_size=batch_size)

    print("Done upserting vector store")


ensure_index()
run_preprocess_and_upsert()

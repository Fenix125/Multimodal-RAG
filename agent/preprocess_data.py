import torch
import numpy as np
from itertools import islice
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, Dataset
from agent.config import CFG
from agent.vector_store import (
    NAMESPACE_POSTER,
    NAMESPACE_TEXT,
    ensure_index,
    batch_upsert,
)

DS_NAME = "stzhao/movie_posters_100k_controlnet"

def get_device_name():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = get_device_name()


def clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    return model, processor

def text_embedding(model, processor, texts, batch_size=64):
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = processor(text=batch, images=None, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        text_emb = model.get_text_features(**tokens)
        out.append(text_emb.detach().cpu().numpy())
    return np.vstack(out)

def img_embedding(model, processor, imgs, batch_size=64):
    out = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        tokens = processor(images=batch, text=None, return_tensors="pt").to(DEVICE)
        img_emb = model.get_image_features(**tokens)
        out.append(img_emb.detach().cpu().numpy())
    return np.vstack(out)


def run_preprocess_and_upsert():
    sample_size = 1000
    batch_size = 64
    
    print(f"Loading dataset ({sample_size} samples): {DS_NAME}")
    stream = load_dataset(DS_NAME, split="train", streaming=True)
    sample = list(islice(stream, sample_size))
    
    movie_ds = Dataset.from_list(sample)
    movie_ds = movie_ds.select_columns(["id", "image", "title", "genres", "overview"])

    ids = [str(row["id"]) for row in movie_ds]

    texts = [row["overview"] for row in movie_ds]
    posters = [row["image"] for row in movie_ds]
    

    print("Loading CLIP model")
    model, processor = clip()
    
    print("Computing text embeddings")
    text_vecs = text_embedding(model, processor, texts, batch_size)

    print("Computing image embeddings")
    img_vecs = img_embedding(model, processor, posters, batch_size)

    metadata = []
    for i in range(len(ids)):
        metadata.append(
            {
                "id": ids[i],
                "title": movie_ds[i]["title"],
                "genres": [genre["name"] for genre in movie_ds[i]["genres"]],
                "overview": movie_ds[i]["overview"],
            }
        )
    
    text_payload = [
        (ids[i], text_vecs[i].tolist(), metadata[i])
        for i in range(len(ids))
    ]
    batch_upsert(text_payload, namespace=NAMESPACE_TEXT, batch_size=batch_size)

    img_payload = [
        (ids[i], img_vecs[i].tolist(), metadata[i])
        for i in range(len(ids))
    ]
    batch_upsert(img_payload, namespace=NAMESPACE_POSTER, batch_size=batch_size)

    print("Done upserting vector store")


ensure_index()
run_preprocess_and_upsert()
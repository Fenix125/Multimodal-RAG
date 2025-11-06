import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel

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

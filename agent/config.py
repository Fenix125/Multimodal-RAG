import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.hugging_face_api_key = os.getenv("HUB_HG_TOKEN", "")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")

CFG = Config()

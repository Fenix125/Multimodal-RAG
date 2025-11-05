from agent.config import CFG
from pinecone import Pinecone, ServerlessSpec


INDEX_NAME = "movie-store"

NAMESPACE_POSTER = "poster"
NAMESPACE_TEXT = "movie-info"

class PC:
    _client = None

    @classmethod
    def client(cls) -> Pinecone:
        if cls._client is None:
            if not CFG.pinecone_api_key:
                raise RuntimeError("PINECONE_API_KEY is missing in environment.")
        cls._client = Pinecone(api_key=CFG.pinecone_api_key)
        return cls._client


def ensure_index(dim: int = 512) -> None:
    pc = PC.client()
    existing = {index["name"] for index in pc.list_indexes()}
    if INDEX_NAME not in existing:
        pc.create_index(
            name=INDEX_NAME,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
        print(f"Created vector store: {INDEX_NAME}")
        return
    print(f"Vector store {INDEX_NAME} already exists")

def index_handle():
    pc = PC.client()
    return pc.Index(INDEX_NAME)


def batch_upsert(vectors, namespace, batch_size = 64) -> None:
    idx = index_handle()
    for i in range(0, len(vectors), batch_size):
        chunk = vectors[i:i+batch_size]
        idx.upsert(vectors=chunk, namespace=namespace)


def query(vector, namespace, top_k = 3):
    idx = index_handle()
    return idx.query(
        vector=vector,
        namespace=namespace,
        top_k=top_k,
        include_values=False,
        include_metadata=True,
    )

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.movies.templates import Movie


class MovieChromaIndexer:
    """
    Chroma-based indexer for multimodal movie retrieval.

    Collections:
      - movies_overviews_text: overview chunks as documents
      - movies_posters_images: poster images as vectors

    Metadata base fields (for both):
      - id          (movie_id)
      - title
      - genres
      - overview
      - release date
      - image       (poster_path, if any)

    Extra text metadata:
      - chunk_index
      - num_chunks

    Extra image metadata:
      - image_id
      - image_path
    """
    def __init__(
        self, text_embedder: TextEmbedder, image_embedder: ImageEmbedder, chunk_size: int = 1024, chunk_overlap: int = 256) -> None:
        self.client = chromadb.PersistentClient(path=config.chroma_path)
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder

        self.movies_text = self.client.get_or_create_collection(
            name="movies_overviews_text",
            metadata={"hnsw:space": "cosine"},
        )
        self.movies_images = self.client.get_or_create_collection(
            name="movies_posters_images",
            metadata={"hnsw:space": "cosine"},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def base_movie_metadata(self, movie: Movie) -> Dict[str, Any]:
        genres_str = ", ".join(movie.genres) if movie.genres else None
        return {
            "movie_id": movie.movie_id,
            "title": movie.title,
            "genres": genres_str,
            "overview": movie.overview,
            "poster_url": movie.poster_url or "",
            "release_date": movie.release_date or "",
        }

    def index(self, movies: List[Movie]) -> None:
        """
        Index both movie overview text and poster images into their respective
        Chroma collections.
        """
        text_ids: List[str] = []
        text_docs: List[str] = []
        text_metadatas: List[Dict[str, Any]] = []

        image_ids: List[str] = []
        image_metadatas: List[Dict[str, Any]] = []

        for movie in movies:
            base_meta = self.base_movie_metadata(movie)

            body = (movie.overview or "").strip()
            if body:
                chunks = self.text_splitter.split_text(body)
                num_chunks = len(chunks)
                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{movie.movie_id}:chunk_{idx}"
                    text_ids.append(chunk_id)
                    text_docs.append(chunk)
                    meta = dict(base_meta)
                    meta.update(
                        {
                            "chunk_index": idx,
                            "num_chunks": num_chunks,
                        }
                    )
                    text_metadatas.append(meta)

            if movie.poster_url:
                image_id = f"{movie.movie_id}_poster"
                image_ids.append(image_id)
                meta = dict(base_meta)
                meta.update(
                    {
                        "image_id": image_id,
                        "image_url": movie.poster_url,
                    }
                )
                image_metadatas.append(meta)

        if text_ids:
            print(f"[INFO] Embedding {len(text_docs)} overview text chunks...")
            text_embeddings = self.text_embedder.embed_documents_to_list(text_docs)
            print("[INFO] Adding text chunks to Chroma collection 'movies_overviews_text'...")
            self.movies_text.add(
                ids=text_ids,
                documents=text_docs,
                metadatas=text_metadatas,
                embeddings=text_embeddings,
            )
        else:
            print("[INFO] No overview text chunks to index.")

        if image_ids:
            print(f"[INFO] Embedding {len(image_metadatas)} poster images...")
            image_paths = [m["image_url"] for m in image_metadatas]
            image_embeddings = self.image_embedder.embed_images_to_list(image_paths)
            print("[INFO] Adding poster images to Chroma collection 'movies_posters_images'...")
            self.movies_images.add(
                ids=image_ids,
                metadatas=image_metadatas,
                embeddings=image_embeddings,
            )
        else:
            print("[INFO] No poster images to index.")

        print("[DONE] Chroma indexing complete.")

    def search_text(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Similarity search over overview text chunks.
        """
        if not query.strip():
            return []

        query_emb = self.text_embedder.embed_queries_to_list([query])
        results = self.movies_text.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for idd, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            hits.append(
                {
                    "source": "text",
                    "id": idd,
                    "distance": dist,
                    "document": doc,
                    "metadata": meta,
                }
            )
        return hits

    def search_images(self, image_query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Similarity search over poster images using a text query describing the poster.
        """
        if not image_query or not image_query.strip():
            return []

        query_emb = self.image_embedder.embed_texts_to_list([image_query])
        results = self.movies_images.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for idd, meta, dist in zip(ids_list, metas_list, dists_list):
            hits.append(
                {
                    "source": "image",
                    "id": idd,
                    "distance": dist,
                    "document": None,
                    "metadata": meta,
                }
            )
        return hits

    def search_images_by_image(self, image_path: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Similarity search over poster images using a provided local image path.
        """
        if not image_path:
            return []

        query_emb = self.image_embedder.embed_images_to_list([image_path])
        results = self.movies_images.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for idd, meta, dist in zip(ids_list, metas_list, dists_list):
            hits.append(
                {
                    "source": "image",
                    "id": idd,
                    "distance": dist,
                    "document": None,
                    "metadata": meta,
                }
            )
        return hits

    def top_text_chunk_for_movie(self, movie_id: str, text_query: Optional[str] = None) -> Optional[str]:
        """
        Return a representative text chunk for a given movie.

        If text_query is provided, we search within that movie's chunks and pick
        the best match. Otherwise we just return the first chunk.
        """
        if text_query:
            query_emb = self.text_embedder.embed_queries_to_list([text_query])
            res = self.movies_text.query(
                query_embeddings=query_emb,
                n_results=1,
                where={"movie_id": movie_id},
                include=["documents", "distances"],
            )
            docs = res.get("documents", [[]])[0]
            return docs[0] if docs else None

        res = self.movies_text.get(
            where={"movie_id": movie_id},
            limit=1,
            include=["documents"],
        )
        docs = res.get("documents") or []
        return docs[0] if docs else None

    def search_multimodal(self, text_query: Optional[str], image_query: Optional[str] = None, k_text: int = 4, k_image: int = 4) -> List[Dict[str, Any]]:
        """
        Combined search:
          - text_query over overview text chunks
          - image_query (short caption/description of desired poster) over poster embeddings

        Returns aggregated results deduped by movie id.
        """
        text_hits = self.search_text(text_query, k=k_text) if text_query else []
        image_hits = self.search_images(image_query, k=k_image) if image_query else []

        combined: Dict[str, Dict[str, Any]] = {}

        def add_hit(hit: Dict[str, Any]) -> None:
            meta = hit["metadata"] or {}
            movie_id = meta.get("movie_id")
            if not movie_id:
                return

            entry = combined.get(movie_id)
            if entry is None:
                genres_meta = meta.get("genres")
                genres = [g.strip() for g in genres_meta.split(",")]

                entry = {
                    "id": movie_id,
                    "title": meta.get("title"),
                    "genres": genres,
                    "overview": meta.get("overview"),
                    "release_date": meta.get("release_date"),
                    "image_paths": [],
                    "text_snippets": [],
                    "min_distance": hit["distance"],
                    "sources": set(),
                }
                combined[movie_id] = entry

            if hit["distance"] < entry["min_distance"]:
                entry["min_distance"] = hit["distance"]

            entry["sources"].add(hit["source"])

            if hit["source"] == "text" and hit.get("document"):
                if hit["document"] not in entry["text_snippets"]:
                    entry["text_snippets"].append(hit["document"])

            if hit["source"] == "image":
                snippet = self.top_text_chunk_for_movie(movie_id, text_query)
                if snippet and snippet not in entry["text_snippets"]:
                    entry["text_snippets"].append(snippet)

            image_path = meta.get("image_url") or meta.get("poster_url")
            if image_path and image_path not in entry["image_paths"]:
                entry["image_paths"].append(image_path)

        for h in text_hits:
            add_hit(h)
        for h in image_hits:
            add_hit(h)

        results = list(combined.values())
        results.sort(key=lambda r: r["min_distance"])

        for r in results:
            r["sources"] = list(r["sources"])

        return results

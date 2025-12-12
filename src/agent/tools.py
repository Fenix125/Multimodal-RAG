from __future__ import annotations

import json
from typing import List, Optional

from langchain_core.tools import tool

from src.vector_db.movie_chroma_store import MovieChromaIndexer

def make_multimodal_search_tool(indexer: MovieChromaIndexer):
    """
    Construct a LangChain tool for multimodal movie search.
    """
    @tool("movie_multimodal_search")
    def movie_multimodal_search(text_query: str, image_query: Optional[str] = None) -> str:
        """
        Search the movie vector index and return structured JSON.

        Args:
            text_query: description of the movie the user is looking for
                (plot, mood, genre, etc.).
            image_query: short caption describing the desired look and feel
                of the movie poster (optional).

        Returns:
            JSON string with 'query' context and 'results' list of movies:
            {
              "query": {"text": "...", "image": "..."},
              "results": [
                {
                  "id": "...",
                  "title": "...",
                  "genres": ["..."],
                  "overview": "...",
                  "sources": ["text", "image"],
                  "text_snippets": ["..."],
                  "image_paths": ["..."],
                  "score": 0.0
                }
              ]
            }
        """
        query_payload = {"text": text_query, "image": image_query}

        if not text_query and not image_query:
            return json.dumps(
                {
                    "query": query_payload,
                    "results": [],
                    "message": "No query provided.",
                }
            )

        results = indexer.search_multimodal(
            text_query=text_query,
            image_query=image_query,
        )

        if not results:
            return json.dumps(
                {
                    "query": query_payload,
                    "results": [],
                    "message": "No relevant movies found.",
                }
            )

        movies: List[dict] = []
        for r in results:
            min_distance = r.get("min_distance")
            score = None
            if min_distance is not None:
                score = 1.0 / (1.0 + float(min_distance))

            movies.append(
                {
                    "id": r.get("id"),
                    "title": r.get("title"),
                    "genres": r.get("genres") or [],
                    "overview": r.get("overview"),
                    "release_date": r.get("release_date"),
                    "sources": r.get("sources") or [],
                    "text_snippets": r.get("text_snippets") or [],
                    "image_paths": r.get("image_paths") or [],
                    "score": score,
                }
            )

        payload = {
            "query": query_payload,
            "results": movies,
        }
        return json.dumps(payload)

    return movie_multimodal_search


def make_image_search_tool(indexer: MovieChromaIndexer):
    """
    Construct a LangChain tool for searching similar movies by poster image.
    """
    @tool("image_search")
    def image_search(image_path: str) -> str:
        """
        Search the poster index using a local image file path.

        Args:
            image_path: path to a local image file (movie poster or similar).

        Returns:
            JSON string with:
              {
                "query": {"image_path": "..."},
                "results": [
                  {
                    "id": "...",
                    "title": "...",
                    "genres": [...],
                    "overview": "...",
                    "sources": ["image"],
                    "text_snippets": ["..."],
                    "image_paths": ["..."],
                    "score": ...
                  }
                ]
              }
        """
        if not image_path:
            return json.dumps(
                {
                    "query": {"image_path": image_path},
                    "results": [],
                    "message": "No image provided.",
                }
            )

        hits = indexer.search_images_by_image(image_path=image_path)
        if not hits:
            return json.dumps(
                {
                    "query": {"image_path": image_path},
                    "results": [],
                    "message": "No matching posters found.",
                }
            )

        combined: List[dict] = []
        for h in hits:
            meta = h.get("metadata") or {}
            min_dist = h.get("distance")
            score = None
            if min_dist is not None:
                score = 1.0 / (1.0 + float(min_dist))

            movie_id = meta.get("movie_id")
            snippet = indexer.top_text_chunk_for_movie(movie_id)

            image_path_meta = meta.get("image_url") or meta.get("image")

            combined.append(
                {
                    "id": movie_id,
                    "title": meta.get("title") or "(untitled movie)",
                    "genres": meta.get("genres") or [],
                    "overview": meta.get("overview"),
                    "release_date": meta.get("release_date"),
                    "sources": ["image"],
                    "text_snippets": [snippet] if snippet else [],
                    "image_paths": [image_path_meta] if image_path_meta else [],
                    "score": score,
                }
            )

        return json.dumps(
            {
                "query": {"image_path": image_path},
                "results": combined,
            }
        )

    return image_search

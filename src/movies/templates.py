from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

class Movie(BaseModel):
    movie_id: str
    title: str
    genres: List[str]
    overview: str
    poster_path: Optional[str] = None

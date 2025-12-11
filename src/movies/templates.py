from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel

class Movie(BaseModel):
    movie_id: str
    title: str
    genres: List[str]
    overview: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None

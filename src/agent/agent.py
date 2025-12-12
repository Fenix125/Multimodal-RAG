from __future__ import annotations
from dataclasses import dataclass

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from src.config import config
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.vector_db.movie_chroma_store import MovieChromaIndexer
from src.agent.prompt import SYSTEM_PROMPT
from src.agent.tools import make_multimodal_search_tool, make_image_search_tool



@dataclass
class Context:
    session_id: str


def build_llm():
    """
    Choose LLM based on config:
      - OpenAI if OPENAI_MODEL is set
      - otherwise Gemini if GEMINI_MODEL is set
    """
    provider = config.llm_provider
    model_name = config.llm_model_name

    if provider is None or model_name is None:
        raise RuntimeError(
            "No LLM configured. Set either OPENAI_MODEL or GEMINI_MODEL in your .env"
        )

    if provider == "openai":
        if not config.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=model_name,
            api_key=config.openai_api_key,
            temperature=0.4,
        )

    if provider == "google":
        if not config.google_ai_api_key:
            raise RuntimeError("GOOGLE_AI_API_KEY is not set.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=config.google_ai_api_key,
            temperature=0.4,
        )

    raise RuntimeError(f"Unknown LLM provider: {provider!r}")



def build_movie_agent():
    """
    Build a LangChain agent with:
      - LLM
      - MovieChromaIndexer-backed multimodal search tools
    """
    text_embedder = TextEmbedder(
        model_name=config.text_embed_model_name,
        device=config.device,
    )
    image_embedder = ImageEmbedder(
        model_name=config.clip_model_name,
        device=config.device,
    )
    indexer = MovieChromaIndexer(
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )

    tools = [
        make_multimodal_search_tool(indexer),
        make_image_search_tool(indexer),
    ]
    checkpointer = InMemorySaver()
    
    agent = create_agent(
        model=build_llm(),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        context_schema=Context,
        checkpointer=checkpointer,
    )
    return agent

from __future__ import annotations

from typing import Dict

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import config
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.vector_db.movie_chroma_store import MovieChromaIndexer
from src.agent.prompt import SYSTEM_PROMPT
from src.agent.tools import make_multimodal_search_tool, make_image_search_tool

HISTORY_STORE: Dict[str, InMemoryChatMessageHistory] = {}


def get_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in HISTORY_STORE:
        HISTORY_STORE[session_id] = InMemoryChatMessageHistory()
    return HISTORY_STORE[session_id]


def build_movie_agent() -> RunnableWithMessageHistory:
    """
    Build a LangChain agent with:
      - LLM
      - MovieChromaIndexer-backed multimodal search tools
      - Chat history via RunnableWithMessageHistory
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

    llm = ChatGoogleGenerativeAI(
        model=config.gemini_model_name,
        temperature=0.2,
        google_api_key=config.google_ai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )

    agent_with_history = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output",
    )
    return agent_with_history

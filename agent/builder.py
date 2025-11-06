from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory, BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool

from agent.config import CFG
from agent.prompt import SYSTEM_PROMPT
from agent.vector_store import query, NAMESPACE_TEXT, NAMESPACE_POSTER
from agent.model import clip, text_embedding, img_embedding
from PIL import Image
from agent.vector_store import index_handle


checkpointer = MemorySaver()


def format_record(data):
    md = data.get("metadata")
    title = md.get("title")
    genres = ", ".join(md.get("genres") or [])
    overview = md.get("overview")
    return f"Title: {title} \nGenre: {genres} \nDescription of movie: {overview}"

@tool("search_text_by_query")
def search_text_by_query(query_text: str) -> str:
    """
    Searches the movie information in a movie's description vector database that are similar to user's query
    Input: description of movie that user would like to find/search similar to
    Returns: relevant movie's titles, genre and description
    """
    model, processor = clip()
    vec = text_embedding(model, processor, [query_text], batch_size=1)[0]
    res = query(vec.tolist(), namespace=NAMESPACE_TEXT, top_k=3)
    items = res.matches or []
    if not items:
        return "No results"
    
    index = index_handle()
    res = index.fetch(
        ids = [it["metadata"]["id"] for it in items],
        namespace=NAMESPACE_POSTER
    )
    poster_paths = []
    for poster_id, poster_obj in res.vectors.items():
        poster_paths.append(poster_obj["metadata"]["poster_path"])

    return f"\n{30*"-"}\n".join(format_record(item) for item in items)


@tool("search_posters_by_query")
def search_posters_by_query(query_text: str) -> str:
    """
    Searches the poster in a movie's poster vector database that are similar to user's text query
    Input: description of movie poster that user would like to find/search similar to
    Returns: relevant movie's titles, genre and description, shows to user their posters
    """
    model, processor = clip()
    vec = text_embedding(model, processor, [query_text], batch_size=1)[0]
    res = query(vec.tolist(), namespace=NAMESPACE_POSTER, top_k=3)
    
    items = res.matches or []
    if not items:
        return "No results"
    
    poster_paths = [it["metadata"]["poster_path"] for it in items]

    index = index_handle()

    movies_text_data = index.fetch(
        ids=[it["metadata"]["id"] for it in items],
        namespace=NAMESPACE_TEXT
    )

    movie_info = []
    for item_id, item_obj in movies_text_data.vectors.items():
        movie_info.append(format_record(item_obj.metadata))

    return f"\n{30*"-"}\n".join(movie_info)

@tool("search_posters_by_image", return_direct=False)
def search_posters_by_image(image_path: str) -> str:
    """
    Searches the poster in a movie's poster vector database that are similar to user's provided poster/image
    Input: image/movie poster path
    Returns: relevant movie's titles, genre and description, shows to user their posters
    """
    model, processor = clip()
    img = Image.open(image_path)
    vec = img_embedding(model, processor, [img], batch_size=1)[0]
    res = query(vec.tolist(), namespace=NAMESPACE_POSTER, top_k=3)
    items = res.matches or []
    if not items:
        return "No results. Try a different image."
    
    poster_paths = [it["metadata"]["poster_path"] for it in items]
    
    index = index_handle()

    movies_text_data = index.fetch(
        ids=[it["metadata"]["id"] for it in items],
        namespace=NAMESPACE_TEXT
    )

    movie_info = []
    for item_id, item_obj in movies_text_data.vectors.items():
        movie_info.append(format_record(item_obj.metadata))

    return f"\n{30*"-"}\n".join(movie_info)

@tool("search_text_by_image")
def search_text_by_image(image_path: str) -> str:
    """
    Searches the movie information in a movie's description vector database that are similar to user's provided poster/image
    Input: image/movie poster path
    Returns: relevant movie's titles, genre and description, shows to user their posters
    """
    model, processor = clip()
    img = Image.open(image_path)
    vec = img_embedding(model, processor, [img], batch_size=1)[0]
    res = query(vec.tolist(), namespace=NAMESPACE_TEXT, top_k=3)
    items = res.matches or []
    if not items:
        return "No results. Try a different image."

    return f"\n{30*"-"}\n".join(format_record(h) for h in items)


def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
        api_key=CFG.google_api_key,
    )
    agent = create_agent(
        model=llm,
        tools=[search_text_by_query, search_posters_by_query, search_posters_by_image, search_text_by_image],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        debug=True
    )
    return agent


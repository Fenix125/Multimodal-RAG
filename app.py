import uuid
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Any

import streamlit as st
from langchain_core.messages import AIMessage, ToolMessage, BaseMessage

from src.agent.agent import build_movie_agent, Context as AgentContext

DEBUG = True

def run_fetch_and_ingest():
    """
    Rebuild the movie index:
      1) Download & preprocess movies into data/processed/movies.jsonl
      2) Index movies into Chroma (overviews + posters)

    Uses the CLI scripts so the app stays in sync with your pipeline.
    """
    try:
        with st.spinner("Step 1/2: downloading & preprocessing movies..."):
            subprocess.run(
                [sys.executable, "-m", "src.scripts.preprocess_movies"],
                check=True,
            )

        with st.spinner("Step 2/2: indexing movies into Chroma..."):
            subprocess.run(
                [sys.executable, "-m", "src.scripts.ingest_chroma_db"],
                check=True,
            )

    except subprocess.CalledProcessError as e:
        st.error(f"Ingestion pipeline failed: {e}")
        return

    st.success("Movie dataset fetched and indexed into Chroma.")


def set_section(name: str):
    st.session_state.section = name


def init_session():
    """
    Initialize session_id, UI messages, and graph history length.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit:{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages: List[dict] = []

    if "history_len" not in st.session_state:
        st.session_state.history_len = 0


def init_agent():
    """
    Lazily initialize and cache the movie agent in Streamlit session_state.
    """
    if "agent" not in st.session_state:
        st.session_state.agent = build_movie_agent()
    return st.session_state.agent


def _normalize_tool_payload(content: Any) -> Any:
    """
    ToolMessage.content can be:
      - dict (ideal)
      - JSON string
      - list of dicts (rare)
    Normalize to a dict with at least optional 'results' key.
    """
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        try:
            data = json.loads(content)
            return data
        except json.JSONDecodeError:
            return None

    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
        return content[0]

    return None


def extract_movies_from_new_messages(new_messages: List[BaseMessage]) -> list:
    """
    Extract movie result payloads from ToolMessages produced in *this* turn.

    We look for tools named like:
      - "movie_multimodal_search"
      - "image_search" (or similar if you reused the name)
    and then expect a payload with a "results" list.
    """
    collected = []

    for msg in new_messages:
        if not isinstance(msg, ToolMessage):
            continue

        tool_name = msg.name or ""

        if tool_name not in {
            "movie_multimodal_search",
            "image_search",
        }:
            continue

        payload = _normalize_tool_payload(msg.content)
        if not payload:
            continue

        if isinstance(payload, dict):
            movies_list = payload.get("results") or []
        elif isinstance(payload, list):
            movies_list = payload
        else:
            movies_list = []

        collected.extend(movies_list)

    return collected


def render_movie_card(movie: dict):
    """
    Stylish card for a single retrieved movie / poster match.
    Shows poster + metadata + relevant snippets.
    """
    title = movie.get("title", "Untitled movie")

    raw_genres = movie.get("genres") or movie.get("genres_str") or []
    if isinstance(raw_genres, str):
        genres = [g.strip() for g in raw_genres.split(",") if g.strip()]
    else:
        genres = list(raw_genres)

    overview = movie.get("overview") or ""
    sources = movie.get("sources", [])
    text_snippets = movie.get("text_snippets", []) or []
    release_date = movie.get("release_date", [])

    image_paths = movie.get("image_paths", []) or []

    score = movie.get("score")

    meta_bits = []
    if genres:
        meta_bits.append("**Genres:** " + ", ".join(genres))
    if score is not None:
        meta_bits.append(f"**Score:** {score:.3f}")
    if release_date:
        meta_bits.append(f"**Release date:** {release_date:}")

    is_image_match = "image" in [s.lower() for s in sources]
    is_text_match = "text" in [s.lower() for s in sources]

    if is_image_match and not is_text_match:
        match_badge = "🎨 Poster-based match"
    elif is_text_match and not is_image_match:
        match_badge = "📄 Overview-based match"
    elif is_image_match and is_text_match:
        match_badge = "🔀 Text + poster match"
    else:
        match_badge = "🔎 Retrieved movie"

    with st.container(border=True):
        st.markdown(
            f"<div style='margin-bottom: 0.5rem; font-size: 0.85rem; opacity: 0.8;'>{match_badge}</div>",
            unsafe_allow_html=True,
        )

        cols = st.columns([1, 2]) if image_paths else [st]
        if image_paths:
            with cols[0]:
                st.image(
                    image_paths[0],
                    caption="Poster",
                    width="stretch",
                )

        right_col = cols[-1]
        with right_col:
            st.markdown(f"### {title}")

            if meta_bits:
                st.markdown(" • ".join(meta_bits))

            if text_snippets:
                st.markdown("**Relevant description snippets:**")
                for snippet in text_snippets:
                    st.markdown(f"> {snippet}")
            elif overview:
                st.markdown("**Overview:**")
                st.markdown(f"> {overview}")

            if is_image_match and len(image_paths) > 1:
                st.markdown("**Other matched posters for this movie:**")
                img_cols = st.columns(min(3, len(image_paths[1:])))
                for idx, path in enumerate(image_paths[1:], start=1):
                    with img_cols[(idx - 1) % len(img_cols)]:
                        st.image(path, caption="", width="stretch")


def render_movies_section(movies, section_title: str = "Retrieved movies"):
    """
    Render a list of movie / poster results with a heading.
    """
    if not movies:
        return

    st.markdown("---")
    st.markdown(f"### {section_title}")

    for movie in movies:
        render_movie_card(movie)


def render_message(msg: dict):
    """
    Render a single message in Streamlit's chat UI.
    """
    role = msg["role"]
    content = msg["content"]
    movies = msg.get("movies") or []

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant" and movies:
            render_movies_section(movies, section_title="Movies used in this answer")


def render_settings_section():
    """
    Settings tab: rebuild the movie index with double confirmation.
    """
    st.subheader("Settings")
    st.caption("Rebuild the local movie index (download + preprocess + index into Chroma).")

    if "confirm_fetch" not in st.session_state:
        st.session_state.confirm_fetch = False

    if not st.session_state.confirm_fetch:
        if st.button("Rebuild movie index", type="primary"):
            st.session_state.confirm_fetch = True
            st.rerun()
    else:
        st.warning(
            "This will re-download a subset of the movie dataset and re-index the database. "
            "Existing Chroma data may be overwritten."
        )
        col_proceed, col_cancel = st.columns([1, 1])
        with col_proceed:
            proceed = st.button("Yes, rebuild", type="primary", use_container_width=True)
        with col_cancel:
            cancel = st.button("Cancel", use_container_width=True)

        if proceed:
            run_fetch_and_ingest()
            st.session_state.confirm_fetch = False
        if cancel:
            st.session_state.confirm_fetch = False
        st.rerun()

def main():
    st.set_page_config(
        page_title="Multimodal Movie Recommender",
        page_icon="🎬",
        layout="wide",
    )
    st.title("Multimodal Movie Recommender")
    st.write(
        "Describe movies you like, ask for similar ones, or upload a poster. "
        "The agent will search over movie overviews and posters."
    )

    init_session()
    agent = init_agent()

    if "section" not in st.session_state:
        st.session_state.section = "Chat"

    st.sidebar.markdown("### Sections")
    st.sidebar.button(
        "Chat",
        use_container_width=True,
        type="primary" if st.session_state.section == "Chat" else "secondary",
        on_click=set_section,
        args=("Chat",),
    )
    st.sidebar.button(
        "Settings",
        use_container_width=True,
        type="primary" if st.session_state.section == "Settings" else "secondary",
        on_click=set_section,
        args=("Settings",),
    )

    section = st.session_state.section

    if section == "Chat":
        for msg in st.session_state.messages:
            render_message(msg)

        with st.sidebar:
            st.markdown("**Poster attachment**")
            uploaded_img = st.file_uploader(
                "📎 Upload a movie poster (optional)",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
            )

        user_input = st.chat_input("Describe a movie you like or ask for recommendations…")

        if user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input, "movies": []}
            )

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    combined_input = user_input

                    if uploaded_img:
                        uploads_dir = Path("data/uploads")
                        uploads_dir.mkdir(parents=True, exist_ok=True)
                        image_path = uploads_dir / uploaded_img.name
                        with open(image_path, "wb") as f:
                            f.write(uploaded_img.getbuffer())

                        combined_input = (
                            f"{user_input}\n"
                            f"[Image path]: {image_path}"
                        )

                    state = agent.invoke(
                        {"messages": [{"role": "user", "content": combined_input}]},
                        config={"configurable": {"thread_id": st.session_state.session_id}},
                        context=AgentContext(session_id=st.session_state.session_id),
                    )

            all_messages: List[BaseMessage] = state["messages"]
            prev_len = st.session_state.history_len
            new_messages = all_messages[prev_len:]
            st.session_state.history_len = len(all_messages)

            ai_msg = next(
                (m for m in reversed(new_messages) if isinstance(m, AIMessage)),
                None,
            )
            output_text = ai_msg.content if ai_msg is not None else "(no output)"
            
            print("[ASSISTANT] > ", all_messages[-1].content)
            print()
            if DEBUG:
                print("---- DEBUG: tool calls this turn ----")
                for msg in reversed(all_messages):
                    if getattr(msg, "type", None) == "human":
                        break

                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        print("[AI] tool calls:")
                        for tc in msg.tool_calls:
                            print("   name:", tc.get("name"), "args:", tc.get("args"))

                    if isinstance(msg, ToolMessage):
                        print("[TOOL]", msg.name, "->", msg.content)
                print("-------------------------------------")
                print()

            movies = extract_movies_from_new_messages(new_messages)

            st.session_state.messages.append(
                {"role": "assistant", "content": output_text, "movies": movies}
            )
            st.rerun()

    elif section == "Settings":
        render_settings_section()


if __name__ == "__main__":
    main()

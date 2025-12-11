import uuid
import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from src.agent.agent import build_movie_agent


def run_fetch_and_ingest():
    """
    Rebuild the movie index:
      1) Download & preprocess movies into data/processed/movies.jsonl
      2) Index movies into Chroma (overviews + posters)
    Uses the CLI scripts so it stays in sync with your pipeline.
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
    Initialize session_id and chat messages for this Streamlit session.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = f"streamlit:{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []


def init_agent():
    """
    Lazily initialize and cache the movie agent in Streamlit session_state.
    """
    if "agent" not in st.session_state:
        st.session_state.agent = build_movie_agent()
    return st.session_state.agent



def extract_movies_from_intermediate_steps(intermediate_steps):
    """
    Pulls movie results from the movie_multimodal_search / image_search tool calls.
    Expects each tool result to be either:
      - a JSON string with {"results": [...]} or
      - a dict with key "results".
    """
    collected = []

    for step in intermediate_steps:
        if not isinstance(step, (list, tuple)) or len(step) != 2:
            continue

        action, result = step
        tool_name = getattr(action, "tool", None)

        if tool_name not in {"movie_multimodal_search", "image_search"}:
            continue

        payload = result
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = None

        if isinstance(payload, dict):
            movies_list = payload.get("results") or []
        elif isinstance(payload, list):
            movies_list = payload
        else:
            movies_list = []

        collected.extend(movies_list)

    return collected


def render_movie_card(movie):
    """
    Stylish card for a single retrieved movie / poster match.
    Shows poster + metadata + relevant snippets.
    """
    title = movie.get("title", "Untitled movie")
    genres = movie.get("genres") or []
    overview = movie.get("overview") or ""
    sources = movie.get("sources", [])
    text_snippets = movie.get("text_snippets", []) or []
    image_paths = movie.get("image_paths", []) or []
    score = movie.get("score")

    meta_bits = []
    if genres:
        meta_bits.append("**Genres:** " + ", ".join(genres))
    if score is not None:
        meta_bits.append(f"**Score:** {score:.3f}")

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


def render_message(msg):
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
                    image_path = None

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

                    res = agent.invoke(
                        {"input": combined_input},
                        config={"configurable": {"session_id": st.session_state.session_id}},
                    )

            output_text = res.get("output", "(no output)")
            intermediate_steps = res.get("intermediate_steps", [])
            movies = extract_movies_from_intermediate_steps(intermediate_steps)

            st.session_state.messages.append(
                {"role": "assistant", "content": output_text, "movies": movies}
            )
            st.rerun()

    elif section == "Settings":
        render_settings_section()


if __name__ == "__main__":
    main()

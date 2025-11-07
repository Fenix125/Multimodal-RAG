import os
import uuid
import json
import gradio as gr
from agent.builder import build_agent
from PIL import Image
from langchain_core.messages import AIMessage

print("Initializing agent:")
agent = build_agent()
print("Multimodal Movie Search Agent ready.")

def new_session():
    return f"gradio:{uuid.uuid4()}"

POSTERS_DIR = os.path.abspath("data/posters")

def process_query(user_input, image, chat_history, session_id):
    try:
        if not user_input and image is None:
            return chat_history, [], session_id

        if image is not None:
            image_path = f"tmp_{uuid.uuid4().hex}.jpg"
            image.save(image_path)
            prompt = f"Find movies similar to this image: {image_path}"
        else:
            prompt = user_input

        response_text, poster_paths = "", []
        result = agent.invoke(
            {"input": prompt},
            config={"configurable": {"session_id": session_id}},
        )

        output = result.get("output")
        if isinstance(output, AIMessage):
            content = output.content
        else:
            content = output
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if "data" in block and isinstance(block["data"], dict):
                        data = block["data"]
                        response_text = data.get("text", "")
                        poster_paths = data.get("posters", [])
                    elif "text" in block:
                        response_text += block["text"] + "\n"

        elif isinstance(content, dict):
            response_text = content.get("text", "")
            poster_paths = content.get("posters", [])
        else:
            response_text = str(content)
        # print("posters from agent:", poster_paths)
        served_posters = []
        for p in poster_paths:
            abs_path = os.path.abspath(p)
            if os.path.exists(abs_path):
                caption = os.path.splitext(os.path.basename(p))[0]
                served_posters.append((abs_path, caption))
            else:
                print(f"not found: {abs_path}")
        # chat_history.append((prompt, response_text.strip()))
        chat_history.append({"role": "user", "content": prompt})
        chat_history.append({"role": "assistant", "content": response_text.strip()})
        return chat_history, served_posters, session_id, ""
    except Exception as e:
        print("process_query error:", e)
        chat_history.append((user_input or "[no input]", f"⚠️ Error: {e}"))
        return chat_history, [], session_id, ""


with gr.Blocks(title="Multimodal Movie Search Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("#Multimodal Movie Search Chat")

    session_id = gr.State(new_session())

    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(label="🖼️ Upload Poster (optional)", type="pil", height=300)
            poster_gallery = gr.Gallery(label="🎞️ Similar Posters", height=400, type="filepath")
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="💬 Gemini Chat", type="messages", render_markdown=False)
            user_input = gr.Textbox(placeholder="Type your query", label="Your query", lines=2)
            send_button = gr.Button("🚀 Send")

    # send_button.click(
    #     fn=process_query,
    #     inputs=[user_input, image_input, chatbot, session_id],
    #     outputs=[chatbot, poster_gallery, session_id],
    # )
    
    send_button.click(
        fn=process_query,
        inputs=[user_input, image_input, chatbot, session_id],
        outputs=[chatbot, poster_gallery, session_id, user_input],  # додано user_input
    )


demo.launch(
    server_name="0.0.0.0",
    server_port=7861,
    allowed_paths=[os.path.abspath("data/posters")]
)

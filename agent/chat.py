from agent.builder import build_agent
from agent.config import CFG
import uuid

def chat():
    session_id = f"cli:{uuid.uuid4()}" 
    print("Building agent...")
    agent = build_agent()
    print("Multimodal Movie Search Agent ready.\n")
    
    while True:
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt == "exit":
            print("bye!")
            break
        config = {"configurable": {"thread_id": session_id}}
        result = agent.invoke({"messages": prompt}, config)
        final_message = result["messages"][-1]
        print(f"> {final_message.content}")


if __name__ == "__main__":
    chat()
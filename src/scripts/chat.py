import asyncio
import uuid

from src.agent.agent import build_movie_agent, Context
from langchain_core.messages import AIMessage, ToolMessage
DEBUG = True

async def main() -> None:
    agent = build_movie_agent()
    session_id = f"cli:{uuid.uuid4()}"
    config = {"configurable": {"thread_id": session_id}}

    print("Chat with agent (type 'exit' to quit).")

    while True:
        user_input = input("[USER] > ")
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            context=Context(session_id=session_id),
        )
        messages = response["messages"]
        ai_msg = messages[-1]

        print("[ASSISTANT] > ", ai_msg.content)
        print()
        
        if DEBUG:
            print("---- DEBUG: tool calls this turn ----")
            for msg in reversed(messages):
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

if __name__ == "__main__":
    asyncio.run(main())

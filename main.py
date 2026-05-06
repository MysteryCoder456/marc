import datetime
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.prompts.chat import MessageLike
from langgraph.checkpoint.memory import InMemorySaver


@tool
def get_time() -> str:
    """
    Get the current time.
    """

    now = datetime.datetime.now()
    return str(now)


def main():
    if not load_dotenv():
        print("Oops... Couldn't load .env file.")

    memory = InMemorySaver()
    agent = create_agent(
        model="openai:gpt-5.4-mini",
        system_prompt="You are a helpful assistant.",
        tools=[get_time],
        checkpointer=memory,
    )

    while True:
        try:
            user_input = input("\n> ")

            response = agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                {"configurable": {"thread_id": "thread-1"}},
                stream_mode="values",
            )
            for chunk in response:
                chunk_msg: MessageLike = chunk["messages"][-1]
                print()
                chunk_msg.pretty_print()

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()

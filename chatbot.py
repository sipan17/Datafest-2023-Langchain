import os

from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from langchain.schema.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder
from langchain.agents.agent_toolkits import create_retriever_tool

from config import config
from embedder import embedding


def main():
    _ = load_dotenv(find_dotenv())  # read local .env file
    
    vectorstore = Chroma(
        persist_directory=os.path.join(config["workdir"], "chroma_dbs", config["model_name_modified"]),
        embedding_function=embedding,
    )
    retriever = vectorstore.as_retriever(**config["retriever_kwargs"])
    
    llm = ChatOpenAI(**config["llm_kwargs"])
    
    tool = create_retriever_tool(
        retriever,
        "search_related_chunks",
        "Searches and returns podcast chunks regarding the question."
    )
    tools = [tool]
    
    memory_key = "chat_history"
    system_message = SystemMessage(
        content=(
            "The author of the podcast is Lex Fridman. "
            "You are a nice chatbot that will help with analyzing his podcast conversations. "
            "Answer as short as possible. "
            "Answer as concise as possible. "
            "If you know the answer beforehand do not use the following info. "
            "If you don't know the answer, feel free to use "
            "the following chunks from podcast conversations to answer the user's question."
        )
    )
    system_message_prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)]
    )
    
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=system_message_prompt)
    
    memory = ConversationBufferMemory(llm=llm, memory_key=memory_key, return_messages=True)
    
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, memory=memory, verbose=True,
        return_intermediate_steps=False
    )
    
    print("Hi, I'm a chatbot created to chat with you about Lex Fridman's podcast")
    print("Type your question or `stop` to terminate")
    while True:
        question = input("Question: ")
        if question == "stop":
            print("Quitting ...")
            break
        result = agent_executor({"input": question})
        print(f"Answer: {result['output']}")


if __name__ == "__main__":
    main()

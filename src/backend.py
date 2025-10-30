import argparse
import glob
import torch

from functools import partial
from pathlib import Path
from typing import Any, List

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from langchain.tools import tool

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings




from src.config import EMBED_MODEL, DB_DIRECTORY, TOKEN_LIMIT, MAX_CHUNKS, HUGGINFACEHUB_API_TOKEN, MODEL_REPO, LLM_PROVIDER

device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {device}")

SYSTEM_PROMPT_TOOL = (
    "You have access to a tool that retrieves context from a relevant document database. "
    "Use the tool to help answer user queries. Answer only based on the retrieved context without making up information. Be concise and to the point."
)

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vector_store = Chroma(
    collection_name="my_collection",
    embedding_function=embedding_model,
    persist_directory=DB_DIRECTORY
)

def get_llm(llm_provider: str, model_repo: str):
    if llm_provider == "huggingface_endpoint":
        llm = HuggingFaceEndpoint(
            repo_id=model_repo,
            task="text-generation",
            huggingfacehub_api_token=HUGGINFACEHUB_API_TOKEN,
        )
        model = ChatHuggingFace(llm=llm)
    elif llm_provider == "huggingface_local":
        llm = HuggingFacePipeline.from_model_id(
            model_id=model_repo,
            task="text-generation",
            device = 0 if torch.cuda.is_available() else -1,
            pipeline_kwargs=dict(
                max_new_tokens=1024,
                do_sample=True,
                repetition_penalty=1.03,
                temperature=0.1,
            ),
        )
        model = ChatHuggingFace(llm=llm)
        
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    return model

def retrieve_context_with_context_limit(query: str, token_limit: int = TOKEN_LIMIT):
    """Retrieve information to help answer a query, limited by token count."""
    retrieved_docs = vector_store.similarity_search(query, k=MAX_CHUNKS)
    
    total_tokens = 0
    selected_docs = []
    content = ""
    
    for doc in retrieved_docs:
        doc.metadata.pop("source")
        serialized = f"Source: {doc.metadata}\nContent: {doc.page_content}"
        
        doc_tokens = len(serialized) // 4 # simple estimation: 1 token ~= 4 characters
        
        if total_tokens + doc_tokens <= token_limit:
            selected_docs.append(doc)
            content += f"\n\n{serialized}"
            total_tokens += doc_tokens
        else:
            break

    return content, selected_docs

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    return retrieve_context_with_context_limit(query, token_limit=TOKEN_LIMIT)

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    last_query = request.state["messages"][-1].text
    content, _ = retrieve_context_with_context_limit(last_query, token_limit=TOKEN_LIMIT)

    system_message = (
        "You are a helpful assistant. Be concise and to the point. Use the following context in your response and do not make up information:"
        f"\n\n{content}"
    )

    return system_message

class SimpleRAG:
    def __init__(self, llm_provider, model_repo, verbose: bool = False):
        self.verbose = verbose
        model = get_llm(llm_provider, model_repo)
        self.agent = create_agent(model, tools=[], middleware=[prompt_with_context])

    def answer_query(self, query: str) -> Any:
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            ):
            if self.verbose:
                event["messages"][-1].pretty_print()
        
        response = event["messages"][-1].content
        splits = response.split("<|im_start|>")
        agent_response = splits[-1].replace("<|im_end|>", "").strip()
        final_answer = agent_response.split("</think>")[-1].strip()
        
        return final_answer
    
    
class SimpleRAGWithToolCalling:
    def __init__(self, llm_provider, model_repo, tools, system_prompt=SYSTEM_PROMPT_TOOL, verbose: bool = False):
        self.verbose = verbose
        model = get_llm(llm_provider, model_repo)
        self.agent = create_agent(model, tools, system_prompt=system_prompt)

    def answer_query(self, query: str) -> Any:
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
            ):
            if self.verbose:
                event["messages"][-1].pretty_print()
        
        response = event["messages"][-1].content
        splits = response.split("<|im_start|>")
        agent_response = splits[-1].replace("<|im_end|>", "").strip()
        final_answer = agent_response.split("</think>")[-1].strip()
        
        return final_answer


# class RAGWithHistory:
#     def __init__(self, llm_provider, model_repo, tools, system_prompt=SYSTEM_PROMPT_TOOL):
#         model = get_llm(llm_provider, model_repo)
#         self.agent = create_agent(model, tools, system_prompt=system_prompt)
#         self.chat_history: List[Document] = []
        
#     def answer_query(self, query: str) -> Any:
#         for event in self.agent.stream(
#             {"messages": [{"role": "user", "content": query}], "chat_history": self.chat_history},
#             stream_mode="values",
#             ):
#             event["messages"][-1].pretty_print()
        
#         self.chat_history.append(Document(page_content=query, metadata={"role": "user"}))

    
if __name__ == "__main__":

    rag_agent = SimpleRAG(
        llm_provider=LLM_PROVIDER,
        model_repo=MODEL_REPO,
    )

    while True:
        user_input = str(input("write a msg :\n"))
        
        if user_input == 'quit' or user_input =='q':
            print('shutting down')
            break
        
        answer = rag_agent.answer_query(user_input+"\n\n")    
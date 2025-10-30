import streamlit as st

from src.backend import SimpleRAG
from src.config import LLM_PROVIDER, MODEL_REPO

if 'rag' not in st.session_state:
    st.session_state.rag = SimpleRAG(
        llm_provider=LLM_PROVIDER,
        model_repo=MODEL_REPO,
    )


st.title("RAG Demo")


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    answer = st.session_state.rag.answer_query(prompt)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.write(answer)
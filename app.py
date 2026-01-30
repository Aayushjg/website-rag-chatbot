import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

st.title("Website Chatbot")

url = st.text_input("Enter Website URL")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def get_text(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script","style","header","footer","nav"]):
        tag.extract()
    return soup.get_text()

if st.button("Index"):
    text = get_text(url)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.success("Indexed!")

q = st.text_input("Ask a question")

if q and "index" in st.session_state:
    q_emb = model.encode([q])
    D,I = st.session_state.index.search(np.array(q_emb),k=3)
    retrieved = " ".join([st.session_state.chunks[i] for i in I[0]])
    if len(retrieved.strip()) < 20:
        ans = "The answer is not available on the provided website."
    else:
        ans = retrieved[:500]
    st.session_state.chat_history.append(("You", q))
    st.session_state.chat_history.append(("Bot", ans))

for role, msg in st.session_state.chat_history:
    st.write(f"**{role}:** {msg}")

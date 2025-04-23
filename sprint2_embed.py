from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st

@st.cache_resource
def get_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks):
    embedder = get_embedder()
    vectors = embedder.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))
    return index, vectors, chunks

import streamlit as st
import numpy as np
from sprint2_embed import get_embedder

def ask_question(index, chunks):
    query = st.text_input("💬 Ask a question about the PDF")
    if query:
        embedder = get_embedder()
        q_vector = embedder.encode([query])
        D, I = index.search(np.array(q_vector), 3)
        retrieved = [chunks[i] for i in I[0]]
        context = "\n".join(retrieved)

        st.write("📚 Retrieved Context:")
        st.write(context)
        return query, context
    return None, None

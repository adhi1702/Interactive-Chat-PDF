import streamlit as st
from streamlit_lottie import st_lottie
import requests
from sprint1_upload import upload_and_extract
from sprint2_embed import chunk_text, create_vector_store
from sprint3_chat import ask_question
from sprint4_groq import call_groq_llm

# Page Config
st.set_page_config(page_title="Chat with PDF", layout="wide")

# Custom Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins&display=swap');

        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            background-color: #f3f6fc;
        }

        .main {
            padding: 20px;
        }

        .stButton>button {
            background: linear-gradient(to right, #667eea, #764ba2);
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 0.6em 1.5em;
        }

        .chat-box {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# Load Lottie Animation
def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

pdf_lottie = load_lottie("https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# App Title
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📚 Intelligent PDF Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Ask smart questions about your PDFs. Powered by FAISS + Groq</p>", unsafe_allow_html=True)
st.markdown("---")

# Tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Chat", "🤖 Chat History", "ℹ️ About"])

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st_lottie(pdf_lottie, height=300, key="pdf")

    with col2:
        st.subheader("📄 Upload a PDF File")
        pdf_text = upload_and_extract()

    if pdf_text:
        st.success("✅ PDF text extracted successfully.")

        with st.spinner("🔍 Embedding and indexing..."):
            chunks = chunk_text(pdf_text)
            index, _, raw_chunks = create_vector_store(chunks)

        st.success("✅ PDF embedded successfully. Ask your question below.")

        question, context = ask_question(index, raw_chunks)

        if question and context:
            answer = call_groq_llm(context, question)

            # Save to chat history
            st.session_state.chat_history.append((question, answer))

            with st.expander("🤖 View Answer"):
                st.markdown(f"<div class='chat-box'>{answer}</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("### 🗂️ Previous Questions & Answers")

    if st.session_state.chat_history:
        for q, a in reversed(st.session_state.chat_history):
            st.markdown(f"""
                <div style='background-color:#e9f1fb; padding: 12px; border-radius: 8px; margin-bottom: 10px;'>
                    <b>🧑 You:</b> {q}
                </div>
                <div style='background-color:#ffffff; padding: 12px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 5px rgba(0,0,0,0.05);'>
                    <b>🤖 Bot:</b> {a}
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")
    else:
        st.info("No questions asked yet. Start chatting in the first tab!")

with tab3:
    st.markdown("### 📌 About This App")
    st.write("""
    This intelligent chatbot allows you to:
    - Upload any PDF
    - Automatically extract and embed content
    - Ask questions with responses from Groq's LLaMA model

    💡 Built with:
    - **Streamlit**
    - **FAISS**
    - **Groq LLM API**
    """)

    st.markdown("---")
    st.caption("Made with ❤️ by Abirami. Designed for the curious minds.")

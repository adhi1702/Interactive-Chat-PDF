import fitz  # PyMuPDF
import streamlit as st

def upload_and_extract():
    st.title("📄 Chat with your PDF")
    pdf_file = st.file_uploader("Upload your PDF", type="pdf")
    if pdf_file:
        pdf_text = ""
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            for page in doc:
                pdf_text += page.get_text()
        st.success("✅ PDF loaded successfully!")
        return pdf_text
    return None

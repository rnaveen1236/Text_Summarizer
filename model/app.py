import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document
import re
import torch

# -------------------------
# 1. Load model and tokenizer
# -------------------------
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# -------------------------
# 2. Text extraction functions
# -------------------------
def load_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

def load_text_from_txt(file):
    return file.read().decode("utf-8")

def load_text_from_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

def load_text(file, file_type):
    if file_type == "pdf":
        return load_text_from_pdf(file)
    elif file_type == "txt":
        return load_text_from_txt(file)
    elif file_type == "docx":
        return load_text_from_docx(file)
    else:
        return ""

# -------------------------
# 3. Text cleaning
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # remove non-ASCII
    return text.strip()

# -------------------------
# 4. Summarization helpers
# -------------------------
def summarize_chunk(chunk, max_length=150, num_beams=4, length_penalty=2.0):
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # BART supports up to 1024 tokens
    )
    summary_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def recursive_summarize(text, chunk_size=800, overlap=100):
    """
    Summarize text of any length using recursive map-reduce.
    """
    splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)

    # Summarize each chunk
    chunk_summaries = [summarize_chunk(chunk) for chunk in chunks if len(chunk.split()) > 20]
    combined_summary = " ".join(chunk_summaries)

    # If summary still too long, summarize again recursively
    if len(tokenizer.encode(combined_summary)) > 1024:
        return recursive_summarize(combined_summary, chunk_size=500, overlap=50)
    else:
        return combined_summary

# -------------------------
# 5. Streamlit UI
# -------------------------
st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("📚 Document & Text Summarizer")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
input_text = st.text_area("Or paste your text here")

summarize_clicked = st.button("Summarize")

if summarize_clicked:
    if uploaded_file:
        file_type = uploaded_file.name.split(".")[-1].lower()
        raw_text = load_text(uploaded_file, file_type)
    elif input_text:
        raw_text = input_text
    else:
        st.warning("Please upload a file or enter text.")
        st.stop()

    cleaned = clean_text(raw_text)
    st.subheader("Extracted Text (Preview)")
    st.write(cleaned[:1000] + "...")

    with st.spinner("Summarizing..."):
        final_summary = recursive_summarize(cleaned)

    st.subheader("Summary")
    st.write(final_summary)
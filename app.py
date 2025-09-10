# document_summarizer.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
import re
import torch

# 1. Load model and tokenizer
MODEL_NAME = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# 2. Text extraction
def load_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text

# 3. Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# 4. Chunking
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)

# 5. Summarization
def summarize_chunk(chunk, max_length=150, num_beams=4, length_penalty=2.0):
    inputs = tokenizer.encode(chunk, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        num_beams=num_beams,
        length_penalty=length_penalty,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 6. Map-Reduce summarization
def summarize_document(text):
    chunks = chunk_text(text)
    summaries = [summarize_chunk(chunk) for chunk in chunks if len(chunk.split()) > 20]
    combined_summary = " ".join(summaries)
    if len(tokenizer.encode(combined_summary)) > 1024:
        return summarize_chunk(combined_summary)
    else:
        return combined_summary

# 7. Streamlit UI
st.set_page_config(page_title="📚 Document Summarizer", layout="wide")
st.title("📚 Document Summarizer with BART")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
input_text = st.text_area("Or paste your text here")

if st.button("Summarize"):
    if uploaded_file:
        raw_text = load_text_from_pdf(uploaded_file)
    elif input_text:
        raw_text = input_text
    else:
        st.warning("Please upload a file or enter text.")
        st.stop()

    cleaned = clean_text(raw_text)
    st.subheader("📑 Extracted Text (Preview)")
    st.write(cleaned[:1000] + "...")

    with st.spinner("Summarizing..."):
        final_summary = summarize_document(cleaned)

    st.subheader("📝 Summary")
    st.write(final_summary)
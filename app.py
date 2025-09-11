import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from docx import Document
import re
from transformers import LEDTokenizer, LEDForConditionalGeneration

# Load model and tokenizer
model_name = "allenai/led-base-16384"
tokenizer = LEDTokenizer.from_pretrained(model_name)
model = LEDForConditionalGeneration.from_pretrained(model_name)

# Text extraction functions
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

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

# LED summarization
def summarize_chunk_led(chunk, max_length=500, min_length=100):
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=16384
    )
    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Recursive summarization
def recursive_summarize_led(text, chunk_size=5000, overlap=500):
    splitter = CharacterTextSplitter(
        separator=". ",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    chunk_summaries = [summarize_chunk_led(chunk) for chunk in chunks if len(chunk.split()) > 20]
    combined_summary = " ".join(chunk_summaries)

    if len(combined_summary.split()) > 5000:
        return recursive_summarize_led(combined_summary, chunk_size=2500, overlap=250)
    else:
        return combined_summary

# Streamlit UI
st.set_page_config(page_title="Document Summarizer", layout="wide")
st.title("Document & Text Summarizer")

uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
input_text = st.text_area("Or paste your text here")

col1, _ = st.columns([1, 1])
summarize_clicked = col1.button("Summarize")

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

    with st.spinner("Summarizing with AllenAI LED..."):
        final_summary = recursive_summarize_led(cleaned)

    st.subheader("Summary")
    st.write(final_summary)

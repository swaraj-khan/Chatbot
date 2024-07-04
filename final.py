
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
import google.generativeai as genai
import re
import datetime

genai.configure(api_key="AIzaSyCj1kqcEZjM51RHbIASsM7GEvh889CDnb4")
model = genai.GenerativeModel('gemini-1.5-flash')

def sanitize_input(input_text):
    return re.sub(r'[^\w\s,.;!?-]', '', input_text)

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return text

def chunk_text(text, max_chunk_size=512):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def embed_text(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return embeddings

def save_embeddings(embeddings, filename="embeddings.pkl"):
    with open(filename, "wb") as file:
        pickle.dump(embeddings, file)

def load_embeddings(filename="embeddings.pkl"):
    with open(filename, "rb") as file:
        return pickle.load(file)

def find_answer(question, embeddings):
    question_embedding = SentenceTransformer('all-MiniLM-L6-v2').encode([question])[0]
    best_chunk = None
    max_cosine = -1
    for chunk, embedding in embeddings:
        cosine = (question_embedding @ embedding) / (np.linalg.norm(question_embedding) * np.linalg.norm(embedding))
        if cosine > max_cosine:
            max_cosine = cosine
            best_chunk = chunk
    response = model.generate_content(best_chunk if max_cosine >= 0.3 else question)
    return response.text

def log_interaction(question, answer):
    with open("interaction_log.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - Question: {question} | Answer: {answer}\n")

st.set_page_config(page_title="Nokia Chatbot", page_icon=":books:", layout="wide")
st.title("🤖 Nokia Chatbot")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    csp = "<meta http-equiv='Content-Security-Policy' content=\"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline';\">"
    st.markdown(csp, unsafe_allow_html=True)

local_css("style.css")

pdf_file_path = "QandA (1).pdf"
if os.path.exists(pdf_file_path):
    with st.spinner('Processing PDF...'):
        text = extract_text_from_pdf(pdf_file_path)
        chunks = chunk_text(text)
        embeddings = embed_text(chunks)
        save_embeddings(list(zip(chunks, embeddings)))
    st.success('PDF processed successfully!')
else:
    st.error('PDF file not found. Please make sure the file is in the correct path.')

if "history" not in st.session_state:
    st.session_state.history = []

from urllib.parse import quote  # Import at the beginning of the file

def display_chat():
    if st.session_state.history:
        for qa in st.session_state.history:
            encoded_answer = quote(qa['answer'])  # URL-encode the answer
            email_body = f"Subject=Regarding your question: {qa['question']}&body={encoded_answer}"
            mailto_link = f"mailto:?{email_body}"
            st.markdown(f'<div class="qa-container"><div class="question"><strong>Q:</strong> {qa["question"]}</div><div class="answer"><strong>A:</strong> {qa["answer"]} <a href="{mailto_link}" target="_blank">📧</a></div></div>', unsafe_allow_html=True)
            st.markdown('<div class="question-answer-divider"></div>', unsafe_allow_html=True)
    question = st.text_input("Ask a question:", key="new_question")
    if question and (not st.session_state.get('last_question') or st.session_state.last_question != question):
        sanitized_question = sanitize_input(question)
        embeddings = load_embeddings()
        with st.spinner('Searching for answers...'):
            answer = find_answer(sanitized_question, embeddings)
        log_interaction(sanitized_question, answer)
        st.session_state.history.append({"question": sanitized_question, "answer": answer})
        st.session_state.last_question = question
        st.rerun()
        
display_chat()

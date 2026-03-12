import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

# OCR libraries
from pdf2image import convert_from_path
import pytesseract

st.title("📄 AI Document Assistant")

# -----------------------------
# Session State
# -----------------------------

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "bm25_index" not in st.session_state:
    st.session_state.bm25_index = None

if "documents_text" not in st.session_state:
    st.session_state.documents_text = []

if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

# -----------------------------
# Query Normalization
# -----------------------------

def normalize_query(q):
    q = q.strip().lower()

    if len(q.split()) <= 2:
        return f"What is {q} in programming?"

    return q

# -----------------------------
# Cache Check
# -----------------------------

def check_cache(question):

    for q in st.session_state.qa_cache:
        if question.lower() in q.lower() or q.lower() in question.lower():
            return st.session_state.qa_cache[q]

    return None

# -----------------------------
# OCR Function for Images
# -----------------------------

def extract_text_from_images(pdf_path):

    images = convert_from_path(pdf_path)

    ocr_text = ""

    for img in images:
        text = pytesseract.image_to_string(img)
        ocr_text += text + "\n"

    return ocr_text

# -----------------------------
# Upload PDFs
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    current_files = [file.name for file in uploaded_files]

    if current_files != st.session_state.uploaded_file_names:

        st.session_state.uploaded_file_names = current_files

        documents = []

        for uploaded_file in uploaded_files:

            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract normal text
            loader = PyPDFLoader(uploaded_file.name)
            docs = loader.load()

            documents.extend(docs)

            # Extract text from images using OCR
            ocr_text = extract_text_from_images(uploaded_file.name)

            if ocr_text.strip():
                documents.append(
                    Document(
                        page_content=ocr_text,
                        metadata={"source": uploaded_file.name}
                    )
                )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vector_store = FAISS.from_documents(split_docs, embeddings)

        st.session_state.vector_store = vector_store

        corpus = [doc.page_content for doc in split_docs]
        tokenized_corpus = [doc.split() for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        st.session_state.bm25_index = bm25
        st.session_state.documents_text = split_docs

        alert = st.success("✅ Documents processed successfully!")
        time.sleep(1)
        alert.empty()

# -----------------------------
# Chat History
# -----------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input("Ask something about the documents")

if question and st.session_state.vector_store:

    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    question = normalize_query(question)

    cached_answer = check_cache(question)

    if cached_answer:

        answer_text = cached_answer

    else:

        vector_results = st.session_state.vector_store.max_marginal_relevance_search(
            question,
            k=6,
            fetch_k=20
        )

        tokenized_query = question.split()

        bm25_scores = st.session_state.bm25_index.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:4]

        keyword_results = [st.session_state.documents_text[i] for i in top_indices]

        combined_results = list({
            doc.page_content: doc
            for doc in (vector_results + keyword_results)
        }.values())

        context = ""

        for doc in combined_results:
            context += doc.page_content + "\n"

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

        if "summary" in question or "summarize" in question:

            prompt = f"""
            Summarize the following document clearly.

            Context:
            {context}
            """

        else:

            prompt = f"""
            Answer the question using ONLY the context below.

            Context:
            {context}

            Question:
            {question}
            """

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):
            answer_text = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        else:
            answer_text = str(content)

        st.session_state.qa_cache[question] = answer_text

    with st.chat_message("assistant"):
        st.write(answer_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer_text}
    )
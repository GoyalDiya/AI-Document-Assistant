📄 AI Document Assistant

An intelligent AI-powered document chat application that allows users to upload PDF documents and interact with them through natural language queries. The system uses Hybrid Retrieval-Augmented Generation (Hybrid RAG), semantic search, and OCR to extract knowledge from documents and provide accurate answers.

🚀 Features

📄 Upload Multiple PDFs
🤖 Chat with your documents
🔎 Hybrid Retrieval (Semantic + Keyword Search)
🧠 Retrieval-Augmented Generation (RAG)
🖼 OCR Support for Image-based PDFs
⚡ Semantic Caching to reduce API calls
💬 Chat Interface similar to ChatGPT
🧾 Document Summarization
📚 Vector Database for fast semantic retrieval

🖥️ Application Interface

Below is the final interface of the application:

<img width="1091" height="690" alt="image" src="https://github.com/user-attachments/assets/b9878c2b-633f-4863-b927-33e09a3af759" />


🏗️ System Architecture

The application follows a Hybrid RAG pipeline:

PDF Upload
     ↓
Text Extraction (PyPDFLoader)
     ↓
OCR Processing (for image-based text)
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
Vector Database (FAISS)
     ↓
Hybrid Retrieval
   ├─ Semantic Vector Search (FAISS)
   └─ Keyword Search (BM25)
     ↓
Context Injection
     ↓
Large Language Model (Google Gemini)
     ↓
Answer / Summary Generation

This hybrid retrieval approach improves accuracy by combining semantic understanding with exact keyword matching.




🛠️ Tech Stack
Frontend

Streamlit

AI / NLP

LangChain

Google Gemini

Vector Search

FAISS

Embeddings

Sentence Transformers

all-MiniLM-L6-v2

OCR

Tesseract OCR

pdf2image

Retrieval

Hybrid Retrieval

Semantic Search (FAISS)

Keyword Search (BM25)



⚙️ Installation


1️⃣ Clone the Repository
git clone https://github.com/yourusername/ai-document-assistant.git
cd ai-document-assistant

2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Install System Dependencies

For OCR support:

Mac:

brew install tesseract
brew install poppler

5️⃣ Add Gemini API Key

Create a .env file:

GOOGLE_API_KEY=your_api_key_here

Generate your key from:

https://aistudio.google.com/app/apikey

▶️ Run the Application
streamlit run app.py

The app will start at:

http://localhost:8501


💡 Example Use Cases

• Research paper exploration
• Interview preparation documents
• Legal document analysis
• Company reports analysis
• Knowledge base chatbot
• Study material summarization



🧠 Key Concepts Demonstrated

This project demonstrates several modern AI engineering techniques:

• Retrieval-Augmented Generation (RAG)
• Hybrid Retrieval (Vector + Keyword Search)
• Vector Databases
• Document Chunking
• Semantic Caching
• OCR for Image-based PDFs
• LLM Integration
• Conversational Interfaces



📈 Future Improvements

• Support for more document formats (DOCX, PPT)
• Advanced semantic search
• Document highlighting in answers
• Multi-user session support
• Cloud deployment (AWS / GCP / Azure)

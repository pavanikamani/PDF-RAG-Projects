# PDF-RAG-Projects
# 📄 Basic PDF RAG Bot

This is my first mini project while learning **Generative AI + RAG concepts**.

The idea behind this project was simple:

👉 Upload a PDF  
👉 Ask questions  
👉 Get answers only from that PDF  

So instead of manually reading a document, this bot reads and understands it for you.

---

## 💡 What this project does

- Accepts a PDF file from the user
- Extracts text page by page using `pdfplumber`
- Breaks the text into smaller chunks (chunking)
- Converts chunks into embeddings using EURI API
- Stores embeddings inside FAISS vector database
- When a question is asked:
  - Question is embedded
  - Similar chunks are retrieved
  - Context + question are sent to LLM
- Maintains conversation memory so follow-up questions work

Basically → **PDF + RAG + Memory = Smart document assistant**

---

## 🧠 Concepts I learned here

- Retrieval Augmented Generation (RAG)
- Chunking strategy (overlap for context continuity)
- Vector database basics (FAISS)
- Embedding generation
- Context injection into LLM prompts
- Conversation memory handling
- Streamlit UI basics

---

## ⚙️ Tech stack

- Streamlit → UI
- pdfplumber → PDF text extraction
- FAISS → vector storage
- NumPy → vector operations
- Requests → API calls
- EURI → embeddings + LLM

---

## ▶️ How to run

```bash
pip install -r requirements.txt
python -m streamlit run basic_rag.py

# 🚀 PDF RAG Projects

This repository contains **two mini projects** I built while learning **Retrieval-Augmented Generation (RAG)** and LLM-based application development.

Instead of separate works, they represent my learning progression from a simple idea → extended AI system.

---

## 📄 Basic PDF RAG Bot

My first project focused on answering questions from a **single PDF**.

### 💡 What it does

* Upload a PDF
* Extract text and split into chunks
* Generate embeddings and store in FAISS
* Retrieve relevant context for questions
* Send context + question to LLM
* Maintain basic conversation memory

👉 A simple **chat-with-your-PDF assistant**

---

## 🧠 Multi-Document Search Engine

This is an extension of the basic version that supports **multiple PDFs**.

### 💡 Added capabilities

* Multi-document ingestion
* Semantic search across all files
* Vector similarity visualization
* Conversation memory display
* Improved retrieval flow

👉 Behaves like a **mini AI knowledge search engine**

---

## ⚙️ Tech Stack

Streamlit • pdfplumber • FAISS • NumPy • Requests • EURI API • scikit-learn • matplotlib

---

## ▶️ Run

```bash
pip install -r requirements.txt
python -m streamlit run basic_rag.py
# or
python -m streamlit run multi_document.py
```

---

⭐ These projects helped me understand how document-based AI assistants are built using **RAG, vector databases, and LLMs**.


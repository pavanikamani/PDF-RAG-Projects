import pdfplumber
import numpy as np
import faiss
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from uuid import uuid4

# ==========================
# CONFIG
# ==========================

EURI_API_KEY = "Your API Key"
EURI_CHAT_URL = "https://api.euron.one/api/v1/euri/chat/completions"
EURI_EMBED_URL = "https://api.euron.one/api/v1/euri/embeddings"

# ==========================
# FUNCTIONS
# ==========================

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def get_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }

    res = requests.post(EURI_EMBED_URL, headers=headers, json=payload)

    if res.status_code != 200:
        st.error(f"Embedding API Error: {res.text}")
        return None

    return np.array(
        [d["embedding"] for d in res.json()["data"]],
        dtype="float32"
    )


def build_index(chunks):
    embeddings = get_embeddings(chunks)
    if embeddings is None:
        return None, None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index, embeddings


def search(query, chunks, index, embeddings, top_k=3):
    q_embed = get_embeddings([query])
    if q_embed is None:
        return [], "", None, None

    D, I = index.search(q_embed.astype("float32"), top_k)

    top_chunks = [chunks[i] for i in I[0]]
    retrieved_vectors = [embeddings[i] for i in I[0]]

    return top_chunks, "\n\n".join(top_chunks), q_embed[0], retrieved_vectors


def chat_answer(query, context):
    if "memory" not in st.session_state:
        st.session_state.memory = []

    headers = {
        "Authorization": f"Bearer {EURI_API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": "Answer only from given context."}]
    messages += st.session_state.memory
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    payload = {
        "model": "gpt-4.1-nano",
        "messages": messages,
        "temperature": 0.3
    }

    res = requests.post(EURI_CHAT_URL, headers=headers, json=payload)

    if res.status_code != 200:
        st.error(f"Chat API Error: {res.text}")
        return "API Error"

    reply = res.json()["choices"][0]["message"]["content"]

    st.session_state.memory.append({"role": "user", "content": query})
    st.session_state.memory.append({"role": "assistant", "content": reply})

    return reply


def plot_vectors(vectors, query_vector):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(np.array(vectors + [query_vector]))

    doc_points = reduced[:-1]
    query_point = reduced[-1]

    fig, ax = plt.subplots()
    ax.scatter(doc_points[:, 0], doc_points[:, 1], label="Chunks", alpha=0.6)
    ax.scatter(query_point[0], query_point[1], color="red", label="Query", marker="x")
    ax.legend()

    st.pyplot(fig)

# ==========================
# UI
# ==========================

st.set_page_config(page_title="AI Multi-PDF Search", layout="wide")

st.title("📄 AI-Powered Multi-PDF Search Engine")

uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

query = st.text_input("Ask something about the documents:")

# ==========================
# PROCESS FILES
# ==========================

if uploaded_files:

    all_text = ""

    for file in uploaded_files:
        temp_name = f"temp_{uuid4()}.pdf"
        with open(temp_name, "wb") as f:
            f.write(file.read())

        extracted = extract_text_from_pdf(temp_name)
        all_text += extracted + "\n"

    if not all_text.strip():
        st.error("No readable text found in PDFs.")
    else:
        st.session_state.chunks = split_text(all_text)
        index, embeddings = build_index(st.session_state.chunks)

        if index is not None:
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            st.success("Documents indexed successfully!")

# ==========================
# QUERY
# ==========================

if query and "index" in st.session_state:

    top_chunks, context, q_vector, retrieved_vectors = search(
        query,
        st.session_state.chunks,
        st.session_state.index,
        st.session_state.embeddings
    )

    answer = chat_answer(query, context)

    st.markdown("### ✅ Answer:")
    st.write(answer)

    with st.expander("📜 Top Relevant Chunks"):
        for chunk in top_chunks:
            st.markdown(f"---\n{chunk}")

    with st.expander("📊 Vector Similarity Visualization"):
        if retrieved_vectors and q_vector is not None:
            plot_vectors(retrieved_vectors, q_vector)

    with st.expander("🧠 Conversation Memory"):
        st.json(st.session_state.memory)

# 🌎 Climate Research Assistant

A Retrieval-Augmented Generation (RAG) application built with LangChain, Google Gemini, ChromaDB, and Streamlit for analyzing climate, environmental, air quality, and scientific research papers.

Users can upload a PDF research paper, ask questions about the document, receive answers grounded in the paper's contents, view supporting source passages, and generate structured research summaries.

---

## Features

### 📄 Research Paper Question Answering

Upload a PDF and ask natural language questions such as:

- How does climate change affect school infrastructure?
- What were the major findings of this study?
- How did climate change impact student health?
- What limitations did the researchers identify?

The application:

1. Extracts text from the PDF
2. Splits the document into chunks
3. Creates vector embeddings
4. Stores embeddings in ChromaDB
5. Retrieves relevant sections
6. Uses Gemini to generate answers based only on retrieved content

---

### 🔍 Source-Aware Retrieval

Each answer includes:

- Retrieved source passages
- Page references
- Supporting evidence from the paper

This helps reduce hallucinations and improves transparency.

---

### 📝 Research Paper Summarizer

Generate a structured summary of any uploaded research paper.

The summarizer extracts:

1. Research Question
2. Methods
3. Dataset / Study Context
4. Key Findings
5. Limitations
6. Why the Research Matters

Example use cases:

- Academic literature reviews
- Climate research exploration
- Air quality studies
- Environmental health research

---

### ⚡ Vector Database Caching

To reduce API usage and improve performance:

- PDFs are embedded only once
- Chroma vector databases are cached locally
- Previously processed PDFs are loaded instantly
- Repeated embedding costs are avoided

---

## Tech Stack

| Component | Technology |
|------------|------------|
| LLM | Google Gemini 3.5 Flash |
| Embeddings | Gemini Embeddings |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Frontend | Streamlit |
| PDF Processing | PyPDFLoader |
| Language | Python |

---

## Project Architecture

```text
PDF Upload
    ↓
PyPDFLoader
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
Chroma Vector Store
    ↓
Retriever
    ↓
Relevant Context
    ↓
Gemini LLM
    ↓
Answer / Summary
```

---

## Example Workflow

### Question Answering

```text
Upload PDF
    ↓
Ask Question
    ↓
Retrieve Relevant Chunks
    ↓
Generate Answer
    ↓
Display Sources
```

### Research Summarization

```text
Upload PDF
    ↓
Extract Full Text
    ↓
Gemini Analysis
    ↓
Structured Summary
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/climate-rag-assistant.git
cd climate-rag-assistant
```

### Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create a `.env` File

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

### Run the Application

```bash
streamlit run app/streamlit_app.py
```

---

## Project Structure

```text
climate-rag-assistant/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── gemini_test.py
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── rag_chain.py
│   └── summarizer.py
│
├── data/
│   └── sample_pdfs/
│
├── cached_vectorstores/
│
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Current Limitations

### Embedding API Rate Limits

The project currently uses Gemini Embeddings.

Large PDFs may generate many chunks and can exceed free-tier embedding quotas.

Possible future solutions:

- Local embedding models
- Background indexing jobs
- Persistent cloud vector storage

---

### Retrieval Is Not Perfect

Like all RAG systems, retrieval quality depends on:

- Chunk size
- Chunk overlap
- Embedding quality
- Question phrasing

Some relevant information may not always appear in the top retrieved chunks.

---

## Future Improvements

### Planned Features

- Multi-document search
- Citation generation
- Source highlighting
- Conversation memory
- PDF comparison mode
- Local embedding models
- Research trend extraction
- Climate knowledge graph
- Research paper recommendation engine

---

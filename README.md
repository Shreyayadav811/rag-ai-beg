# RAG PDF Q&A (Prototype)

A Retrieval-Augmented Generation (RAG) pipeline that answers questions from PDF documents. It retrieves the most relevant passages from a PDF and passes them to a local LLM, so answers are grounded in the document instead of the model's memory.

> **Status: prototype / demo.** Built to understand and demonstrate the core RAG pipeline end to end. It works best on text-based PDFs like the included sample. It is not tested across all document types (see [Known Limitations](#known-limitations)).

## How It Works

```
PDF -> extract text per page -> chunk (1000 chars, 200 overlap)
    -> embed chunks (bge-small-en-v1.5) -> store in ChromaDB

Question -> embed query -> top-k similarity search in ChromaDB
         -> build context from retrieved chunks -> local LLM (Ollama) -> answer
```

1. **Load** (`pdf_loader.py`): extracts text page by page using `pypdf`, keeping source and page number as metadata.
2. **Chunk** (`chunker.py`): splits each page into fixed-size character chunks (1000 chars) with 200 chars of overlap so context is not lost at boundaries.
3. **Embed** (`embeddings.py`): converts chunks to normalized vectors with `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
4. **Store** (`vector_store.py`): saves chunks, embeddings and metadata in a persistent ChromaDB collection using cosine similarity.
5. **Retrieve** (`retriever.py`): embeds the question (with the BGE query prefix) and fetches the top-k most similar chunks.
6. **Generate** (`llm.py`, `rag.py`): builds a context string with page numbers and asks a local model through Ollama to answer using only that context, or say it doesn't know.

## Tech Stack

- Python
- ChromaDB (vector store)
- sentence-transformers, `BAAI/bge-small-en-v1.5` (embeddings)
- Ollama with `qwen2.5:1.5b` (local LLM)
- pypdf (PDF text extraction)

**Why Ollama?** I initially experimented with the Groq API for generation, then moved to a local model through Ollama because it is free and runs fully offline.

## Project Structure

```
backend/
  services/
    pdf_loader.py     # PDF -> page text
    chunker.py        # fixed-size overlapping chunks
    embeddings.py     # sentence-transformers embeddings
    vector_store.py   # ChromaDB storage
    retriever.py      # query -> top-k chunks
    llm.py            # Ollama answer generation
    rag.py            # retrieval + generation
  data/documents/     # sample PDF
  test_ingestion.py   # load -> chunk -> embed -> store
  test_retrieval.py   # check retrieval results
  test_rag.py         # end-to-end question answering
```

## Setup

1. Install [Ollama](https://ollama.com) and pull the model:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
2. Install dependencies:
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Ingest the sample document (run from `backend/`):
   ```bash
   python test_ingestion.py
   ```
4. Ask a question:
   ```bash
   python test_rag.py
   ```
   Edit the `question` variable in `test_rag.py` to try your own.

## Known Limitations

- **Text PDFs only.** Scanned or image-based PDFs have no extractable text, so nothing is indexed.
- **Fixed-size chunking.** Chunks are cut by character count and can split sentences, tables or sections.
- **No reranking.** Results rely on vector similarity alone.
- **Small local model.** `qwen2.5:1.5b` is fast but weaker at complex or multi-part questions.
- **Simple ingestion.** Chunk IDs are index-based, so re-ingesting or adding multiple documents needs more careful ID handling.
- **No UI or API yet.** It is run through scripts.

## Possible Improvements

- Sentence- or section-aware chunking
- A reranker on top of retrieval
- OCR for scanned PDFs
- A FastAPI backend and a simple web UI
- Support for multiple documents with unique IDs

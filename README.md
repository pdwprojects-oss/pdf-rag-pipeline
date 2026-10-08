# PDF RAG Pipeline

A small Python project that prepares PDF text for a RAG system.

## Pipeline

PDF → Text extraction → Cleaning → Chunking → Gemini embeddings

- `reader.py`: extracts text from a text-based PDF.
- `cleaning.py`: normalizes whitespace and field formatting.
- `chunking.py`: splits records into chunks with metadata.
- `embedding.py`: generates vectors using Gemini.

This project covers data preparation. Retrieval and answer generation are not implemented yet.

## Requirements

- Python 3.10 or newer
- A Gemini API key
- Internet access for embedding generation

## Setup

Install dependencies from the repository root:

```bash
python -m pip install -r knowledge/requirements.txt
```

Copy `knowledge/.env.example` to `knowledge/.env` and replace the placeholder with your own API key.

## Run

Place `synthetic_medibridge_healthcare_rag_dataset.pdf` inside `knowledge/`, then run:

```bash
python knowledge/reader.py
python knowledge/cleaning.py
python knowledge/chunking.py
python knowledge/embedding.py
```

Outputs are saved inside `knowledge/`.

The input dataset is not included. To use a different filename, update the filename in each script.

## Notes

- The cleaning rules are tailored to the MediBridge dataset format.
- Scanned PDFs require OCR, which is not included.
- Chunks use a maximum of 250 words and a 40-word overlap within long records.
- Embeddings use `gemini-embedding-001`.
- Embedding generation sends chunk text to Google's API.
- API access is subject to quota and billing limits.
- API keys, source documents, and generated outputs are excluded from Git.
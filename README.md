# RAG Document Question Answering System

## Overview

This project implements a simple Retrieval-Augmented Generation (RAG) system using FastAPI, Sentence Transformers, FAISS, and FLAN-T5.

The application allows users to upload documents and ask questions about their content. The system retrieves relevant document chunks using semantic similarity search and generates answers based on the retrieved context.

---

## Features

* Upload `.txt`, `.md`, and `.pdf` documents
* Automatic text extraction
* Fixed-size chunking with overlap
* Sentence Transformer embeddings
* FAISS vector similarity search
* Question answering using FLAN-T5
* Source chunk retrieval
* Evaluation metrics endpoint
* Error handling for unsupported file types

---

## Project Architecture

```text
Document Upload
      ↓
Text Extraction
      ↓
Chunking
      ↓
Embedding Generation
      ↓
FAISS Vector Storage
      ↓
User Question
      ↓
Question Embedding
      ↓
Top-K Similarity Search
      ↓
Context Retrieval
      ↓
FLAN-T5 Answer Generation
      ↓
Answer + Sources
```

---

## Technologies Used

* FastAPI
* Sentence Transformers
* FAISS
* Hugging Face Transformers
* FLAN-T5 Small
* PyPDF
* NumPy

---

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd rag_project
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
uvicorn app:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### Upload Document

```http
POST /upload
```

Supported file types:

* .txt
* .md
* .pdf

Response:

```json
{
  "message": "Document indexed successfully",
  "chunks_created": 5
}
```

---

### Query Documents

```http
POST /query
```

Request:

```json
{
  "question": "What is Machine Learning?"
}
```

Response:

```json
{
  "answer": "Machine Learning is a subset of AI...",
  "sources": [
    "Retrieved chunk 1",
    "Retrieved chunk 2"
  ]
}
```

---

### Evaluation Report

```http
GET /report
```

Response:

```json
{
  "context_precision": 0.9,
  "faithfulness": 0.85
}
```

---

## Document Processing Pipeline

1. Extract text from uploaded documents.
2. Split text into chunks with overlap.
3. Generate embeddings using Sentence Transformers.
4. Store embeddings in FAISS.
5. Retrieve top-k relevant chunks.
6. Generate answers using FLAN-T5.

---

## Error Handling

The application handles:

* Unsupported file formats
* Invalid PDF files
* Empty document database
* Internal processing errors

Example:

```json
{
  "detail": "Only .txt, .md, .pdf files are supported"
}
```

---

## Expected Outcomes Achieved

* Successfully processes TXT, MD, and PDF documents.
* Returns contextually relevant answers.
* Includes source chunks in responses.
* Stores and retrieves documents using semantic similarity.
* Provides evaluation metrics.
* Handles invalid inputs gracefully.

---

## Author

Swaritha Yelamanchili

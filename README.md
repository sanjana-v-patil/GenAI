# Legal Document Question Answering System

## Overview

This project implements a robust **multi-agent system** for **Question Answering (QA)** over legal documents, using modern Natural Language Processing (NLP) and retrieval techniques. It is designed to extract, preprocess, embed, and query legal contracts using fine-tuned transformer models, LoRA/QLoRA optimization, and semantic validation. The system integrates document parsing, text chunking, embedding storage (via ChromaDB), and a multi-model retrieval-augmented generation (RAG) framework.

> **Note**: For testing and evaluation purposes, four legal documents were used from the **CUAD\_v1 dataset** (Contract Understanding Atticus Dataset v1), specifically various versions of a "Services Agreement" from Federated Government Income Securities Inc.

---

## Features

* **PDF Text Extraction** with OCR fallback
* **Advanced Sentence and Chunk Tokenization** using spaCy and TikToken
* **Embedding with SentenceTransformers** for vector-based semantic retrieval
* **Re-ranking with CrossEncoder** for improved passage relevance
* **Fine-tuned QA Models** (RoBERTa with LoRA/QLoRA and PEFT support)
* **Persistent Vector Store** using ChromaDB
* **Contextual Memory Agent** for handling multi-turn conversation history
* **Answer Validation Agent** using semantic similarity, NER overlap, and legal term detection
* **Optional Summarization Agent** for generating concise document or answer summaries
* **Few-shot QA Handling** for common question patterns
* **Gemini API Integration** for content generation from aggregated contexts

---

## Installation

Ensure Python 3.8+ is installed. Then install the dependencies:

```bash
pip install pymupdf regex spacy tiktoken torch chromadb numpy \
    google-generativeai sentence-transformers scikit-learn transformers \
    datasets bitsandbytes accelerate peft pytesseract pillow bert-score
python -m spacy download en_core_web_sm
```

---

## Directory Structure

```
.
├── code.py                    # Main system code (modular multi-agent architecture)
├── README.md                  # Project documentation (this file)
└── /content/                  # Sample legal PDFs used for testing (CUAD dataset)
```

---

## System Architecture

The system is modular and composed of several agents:

| Agent                    | Responsibility                                       |
| ------------------------ | ---------------------------------------------------- |
| `DocumentExtractorAgent` | Extracts raw text from PDF files                     |
| `TextProcessingAgent`    | Chunks and tokenizes text for embedding              |
| `EmbeddingAgent`         | Computes dense vector embeddings                     |
| `DatabaseAgent`          | Manages ChromaDB for semantic search                 |
| `QueryAgent`             | Executes queries using RAG, re-ranking, and QA model |
| `MemoryAgent`            | Maintains conversational context                     |
| `AnswerValidatorAgent`   | Scores and refines answers                           |
| `SummaryGeneratorAgent`  | Optionally summarizes long answers or documents      |
| `CoordinatorAgent`       | Orchestrates all other agents and system logic       |

---

## Usage

### 1. **Initialization**

To start the system and initialize with legal PDFs:

```python
coordinator = CoordinatorAgent()
coordinator.initialize_system(pdf_files)
```

The following CUAD documents were used for testing:

```python
pdf_files = [
    "FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT.pdf",
    "FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_AMENDMENT.pdf",
    "FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_POWEROF.pdf",
    "FEDERATEDGOVERNMENTINCOMESECURITIESINC_04_28_2020-EX-99.SERV AGREE-SERVICES AGREEMENT_SECONDAMENDMENT.pdf",
]
```

---

### 2. **Fine-Tuning**

You can fine-tune the QA model on your domain-specific dataset using:

```python
training_examples = [
    {
        "question": "What is consideration in contract law?",
        "context": "Consideration is value exchanged between parties to a contract...",
        "answer": "value exchanged between parties to a contract"
    }
]
coordinator.fine_tune_qa_model(training_examples)
```

---

### 3. **Interactive QA**

Once initialized, the system accepts user queries and provides semantically validated answers:

```python
answer, metrics = coordinator.process_query("Who is the agreement between?")
```

The output includes:

* Refined Answer
* Semantic Similarity Score
* Legal Entity Overlap Count
* Legal Term Presence
* Validation Score and Reason

---

### 4. **Image OCR (Optional)**

You can also extract text from images using:

```python
ocr_text = extract_text_from_image("path/to/image.png")
answer, metrics = coordinator.process_query(ocr_text)
```

---

## Model Configuration

* QA Model: `deepset/roberta-base-squad2`
* Embedding Model: `sentence-transformers/all-MiniLM-L6-v2`
* Re-ranker: `cross-encoder/ms-marco-MiniLM-L-12-v2`
* Summarizer (optional): `sshleifer/distilbart-cnn-12-6`
* PEFT: LoRA or QLoRA (configurable)

---

## Evaluation and Validation

Each answer is scored and validated using:

* Cosine similarity between question and answer
* Named entity overlap
* Legal term presence
* Length and specificity checks

A final composite score determines answer validity.

---

## Gemini API Integration

If a Gemini API key is provided, the system will generate enhanced answers from document context. Set the key via:

```python
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

## Legal Dataset Notice

The system was tested on four legal contract documents sourced from the [CUAD\_v1 dataset](https://www.atticusprojectai.org/cuad), which is a publicly available benchmark for legal document understanding.

---

## License

This project is released for research and educational purposes. Ensure compliance with all applicable terms when using pretrained models and datasets.

---

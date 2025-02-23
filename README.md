# RAG Implementation with Cohere and LangChain

This project implements a Retrieval Augmented Generation (RAG) system using Cohere's language models and LangChain framework. The system can answer questions about HR policies by referencing a knowledge base.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Features](#features)
- [Usage](#usage)
- [Configuration](#configuration)
- [How It Works](#how-it-works)

## Prerequisites

- Python 3.10 or higher
- Cohere API key
- Required Python packages

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Chains
```

2. Install required packages:
```bash
pip install langchain-cohere langchain-community chromadb pydantic python-dotenv
```

3. Set up environment variables:
Create a `.env` file in the project root:
```bash
COHERE_API_KEY=your_api_key_here
```

## Project Structure

```
Chains/
├── data/
│   ├── globalcorp_hr_policy.txt
│   ├── local_vectorstore/
│   └── local_docstore/
├── src/
│   └── CohereQnA.py
└── README.md
```

## Features

- **Document Loading**: Automatic processing of HR policy documents
- **Vector Embeddings**: Powered by Cohere's embedding models
- **Local Storage**: Using Chroma for vector storage
- **Smart Retrieval**: Parent-child document architecture
- **Interactive QA**: Question answering using RAG

## Usage

1. Place your HR policy document in `data/globalcorp_hr_policy.txt`

2. Run the QnA system:
```bash
python src/CohereQnA.py
```

3. The system will:
   - Load and process the document
   - Create embeddings
   - Store them in a local vector store
   - Answer questions about the HR policy

## Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Parent chunk size | 1000 | Characters per parent chunk |
| Child chunk size | 200 | Characters per child chunk |
| Overlap | 20 | Characters overlapping between chunks |
| Model | command-light | Cohere model used |
| Temperature | 0 | Deterministic output setting |

## How It Works

### Parent-Child Document Retrieval

The system uses a two-tier document splitting approach:

- **Child Splitter**: Creates small, precise chunks for accurate matching
- **Parent Splitter**: Maintains larger chunks for context preservation

Example scenario:
```
Document Structure:
├── Parent Chunk (Chapter-sized, 1000 chars)
│   └── Child Chunks (Paragraph-sized, 200 chars)
```

## Key-Value Document Store

The system uses a key-value document store to maintain relationships between document chunks and their sources. This is implemented using LangChain's storage system:

### Features

- **Relationship Preservation**: Maintains links between parent and child document chunks
- **Source Tracking**: Associates chunks with their original source documents
- **Efficient Retrieval**: Enables quick lookup of parent documents when child chunks are retrieved

### Benefits

- **Precise Answers**: Find exact relevant passages
- **Context Preservation**: Access broader context when needed
- **Balanced Retrieval**: Optimal mix of specificity and context
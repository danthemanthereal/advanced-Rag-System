# advanced-Rag-System

## Overview

This project implements an **advanced Retrieval-Augmented Generation (RAG) pipeline** for experimenting with different retrieval and ranking approaches.

The pipeline is designed to be **modular and easily extensible**. Additional components such as **query augmentation**, **recursive chunking**, or **semantic chunking** can be integrated into the pipeline in the future.

Currently, these extensions are **not implemented** and the project focuses on the core retrieval pipeline.

## How to Start

### 1. Create a virtual environment and activate

```bash
python -m venv venv

source venv/bin/activate
```

### 2. Install the requirements

```bash

pip install -r requirements.txt
```

### 3. Load the data and create indicies 

```bash

cd scripts folder and execute:

python load_bier_dataset.py

python create_bm25_index.py

python create_vector_db_index.py

```

### 4. Start the RAG pipeline

```bash

move to project folder and execute: python main.py

```

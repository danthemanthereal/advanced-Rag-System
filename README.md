# advanced-Rag-System

## Overview

This project implements an **advanced Retrieval-Augmented Generation (RAG) pipeline** for experimenting with different retrieval and ranking approaches.

The pipeline is designed to be **modular and easily extensible**. Additional components such as **query augmentation**, **recursive chunking**, or **semantic chunking** can be integrated into the pipeline in the future.

Currently, these extensions are **not implemented** and the project focuses on the core retrieval pipeline.

## How to Start

### 1. Create a virtual environment

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt


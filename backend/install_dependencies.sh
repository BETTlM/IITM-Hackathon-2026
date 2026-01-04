#!/bin/bash

# Alternative installation script that installs packages in stages
# This helps avoid dependency resolution issues

echo "Installing dependencies in stages..."

# Stage 1: Core packages
echo "Stage 1: Installing core packages..."
pip install --upgrade pip setuptools wheel
pip install fastapi uvicorn python-dotenv pydantic requests

# Stage 2: LangChain core
echo "Stage 2: Installing LangChain core..."
pip install langchain langchain-core langchain-community

# Stage 3: LangChain integrations
echo "Stage 3: Installing LangChain integrations..."
pip install langchain-ollama langchain-huggingface

# Stage 4: Vector database
echo "Stage 4: Installing vector database..."
pip install chromadb

# Stage 5: LangChain Chroma integration
echo "Stage 5: Installing LangChain Chroma..."
pip install langchain-chroma

# Stage 6: LangGraph
echo "Stage 6: Installing LangGraph..."
pip install langgraph

# Stage 7: Embeddings
echo "Stage 7: Installing embeddings..."
pip install sentence-transformers

echo "Installation complete!"


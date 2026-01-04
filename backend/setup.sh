#!/bin/bash

# Setup script for Visa Benefits System

echo "=========================================="
echo "Visa Benefits System - Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version || { echo "Python 3 not found. Please install Python 3.11+"; exit 1; }

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv ../venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ../venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check Ollama
echo ""
echo "Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "WARNING: Ollama not found. Please install Ollama:"
    echo "  macOS: brew install ollama"
    echo "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    read -p "Press Enter to continue after installing Ollama..."
fi

# Pull required model
echo "Pulling llama3.2 model (this may take a while)..."
ollama pull llama3.2 || { echo "ERROR: Failed to pull model. Is Ollama running?"; exit 1; }

# Ingest data
echo ""
echo "Ingesting benefits data..."
python ingest.py || { echo "ERROR: Data ingestion failed"; exit 1; }

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To start the server:"
echo "  cd backend"
echo "  source ../venv/bin/activate"
echo "  python main.py"
echo ""
echo "To test the API:"
echo "  python test_workflow.py"
echo ""


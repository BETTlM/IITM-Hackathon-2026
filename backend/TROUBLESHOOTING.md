# Troubleshooting Guide

## Dependency Installation Issues

### Problem: "resolution-too-deep" or dependency conflicts

This is a common issue with complex dependency trees, especially with LangChain and ChromaDB.

### Solutions (try in order):

#### Solution 1: Use Staged Installation
```bash
cd backend
chmod +x install_dependencies.sh
./install_dependencies.sh
```

This installs packages in stages, which helps pip resolve dependencies incrementally.

#### Solution 2: Use Python 3.11
Some packages may not fully support Python 3.14 yet. Try using Python 3.11:

```bash
# Create new venv with Python 3.11
python3.11 -m venv ../venv311
source ../venv311/bin/activate
pip install --upgrade pip setuptools wheel
./install_dependencies.sh
```

#### Solution 3: Install Core Packages First
```bash
# Install core packages
pip install fastapi uvicorn python-dotenv pydantic requests

# Then install LangChain
pip install langchain langchain-core langchain-community

# Then install integrations
pip install langchain-ollama langchain-huggingface

# Then ChromaDB
pip install chromadb

# Then LangChain Chroma
pip install langchain-chroma

# Then LangGraph
pip install langgraph

# Finally embeddings
pip install sentence-transformers
```

#### Solution 4: Use Specific Older Versions
If newer versions cause issues, try these known-compatible versions:

```bash
pip install \
  fastapi==0.115.9 \
  uvicorn==0.30.6 \
  langchain==0.3.7 \
  langchain-core==0.3.12 \
  langchain-community==0.3.7 \
  langchain-ollama==0.1.3 \
  langchain-huggingface==0.0.3 \
  langchain-chroma==0.1.2 \
  langgraph==0.2.48 \
  chromadb==0.4.24 \
  sentence-transformers==2.7.0 \
  pydantic==2.9.2
```

#### Solution 5: Use pip-tools
```bash
pip install pip-tools
pip-compile requirements.txt
pip-sync
```

## Ollama Issues

### Problem: "Connection refused" or "Model not found"

**Check Ollama is running:**
```bash
ollama list
```

**If not running:**
```bash
ollama serve
```

**Pull the model:**
```bash
ollama pull llama3.2
```

**Verify:**
```bash
ollama list
# Should show llama3.2
```

## Scipy Import Issues

### Problem: Import hangs or "ModuleNotFoundError: scipy"

This is a common issue where scipy installation is broken or incompatible.

**Solution 1: Use simplified ingestion script**
```bash
cd backend
python ingest_simple.py
```

**Solution 2: Fix scipy installation**
```bash
cd backend
./fix_scipy.sh
python ingest_simple.py
```

**Solution 3: Manual fix**
```bash
pip uninstall -y scipy numpy
pip install numpy==1.26.4 scipy==1.13.1
python ingest_simple.py
```

## ChromaDB Issues

### Problem: "Database not found" or import errors

**Re-run ingestion:**
```bash
cd backend
python ingest_simple.py  # Use simplified version
# OR
python ingest.py  # Standard version
```

**Check database exists:**
```bash
ls -la visa_db/
```

**If issues persist, delete and recreate:**
```bash
rm -rf visa_db/
python ingest.py
```

## Import Errors

### Problem: "ModuleNotFoundError" for langchain packages

**Check installation:**
```bash
pip list | grep langchain
```

**Reinstall specific package:**
```bash
pip install --force-reinstall langchain-ollama
```

## Python Version Issues

### Problem: Packages not compatible with Python 3.14

**Use Python 3.11:**
```bash
# Check available Python versions
python3.11 --version

# Create new venv
python3.11 -m venv ../venv311
source ../venv311/bin/activate
```

## Memory Issues

### Problem: Out of memory during model loading

**Solutions:**
- Use smaller embedding model: Change `all-MiniLM-L6-v2` to `all-MiniLM-L12-v2` (if available)
- Reduce batch size in ingestion
- Close other applications

## Still Having Issues?

1. **Check Python version**: `python --version` (should be 3.11 or 3.12)
2. **Check virtual environment**: Make sure you're in the venv
3. **Clear pip cache**: `pip cache purge`
4. **Fresh start**: Delete venv and recreate
5. **Check logs**: Look for specific error messages

## Getting Help

If none of these solutions work, please provide:
- Python version: `python --version`
- Error message (full traceback)
- What step failed (installation, ingestion, or running server)
- Operating system: `uname -a`


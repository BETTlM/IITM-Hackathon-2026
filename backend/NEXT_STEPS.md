# Next Steps - Getting Your System Running

## ✅ Current Status
- ✅ Database created (visa_db exists)
- ⚠️  Need to verify packages and Ollama model

## Step-by-Step Setup

### Step 1: Ensure You're in the Right Virtual Environment

```bash
# Make sure you're in your venv
cd /Users/bettim/Documents/Semester\ 4/IIT
source venv/bin/activate  # or: source ../venv/bin/activate

# Verify Python version
python --version  # Should be 3.11
```

### Step 2: Install Remaining Dependencies (if needed)

```bash
cd backend

# Install all requirements
pip install -r requirements.txt

# OR if that fails, use staged installation
./install_dependencies.sh
```

### Step 3: Setup Ollama Model

```bash
# Make sure Ollama is running
ollama serve

# In another terminal, pull the model
ollama pull llama3.2

# Verify
ollama list  # Should show llama3.2
```

### Step 4: Verify Database (Already Done ✅)

Your database exists at `backend/visa_db`. If you need to recreate it:

```bash
cd backend
python ingest_simple.py
```

### Step 5: Start the Server

```bash
cd backend
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Test the API

**In a new terminal:**

```bash
# Test 1: Health check
curl http://localhost:8000/

# Test 2: Get benefits (Student with Classic card)
curl -X POST http://localhost:8000/benefits \
  -H "Content-Type: application/json" \
  -d '{
    "card_number": "4111-****-****-1111",
    "user_context": "student",
    "preferred_language": "en",
    "location": "Chennai"
  }'

# Test 3: Traveler with Infinite card
curl -X POST http://localhost:8000/benefits \
  -H "Content-Type: application/json" \
  -d '{
    "card_number": "4222-****-****-2222",
    "user_context": "traveler",
    "preferred_language": "en"
  }'
```

**Or use the test script:**
```bash
cd backend
python test_workflow.py
```

### Step 7: View API Documentation

Open in browser:
```
http://localhost:8000/docs
```

This gives you an interactive Swagger UI to test the API.

## Quick Verification Checklist

Run these commands to verify everything:

```bash
# 1. Check Python and packages
python --version
python -c "import fastapi, langchain, chromadb; print('✅ Packages OK')"

# 2. Check Ollama
ollama list | grep llama3.2 && echo "✅ Ollama OK" || echo "❌ Need: ollama pull llama3.2"

# 3. Check database
ls -la backend/visa_db && echo "✅ Database OK" || echo "❌ Need: python ingest_simple.py"

# 4. Start server
cd backend && python main.py
```

## Common Issues

### "Module not found"
- Make sure virtual environment is activated
- Run: `pip install -r requirements.txt`

### "Ollama connection error"
- Make sure `ollama serve` is running
- Check: `ollama list`

### "Database not found"
- Run: `python ingest_simple.py`

### "Port 8000 already in use"
- Change port in `main.py`: `uvicorn.run(api, host="0.0.0.0", port=8001)`

## Success Indicators

You're ready when:
- ✅ Server starts without errors
- ✅ `curl http://localhost:8000/` returns JSON
- ✅ `/benefits` endpoint returns benefit recommendations
- ✅ Swagger UI loads at `/docs`

## Next: Demo Scenarios

Once everything works, try these demo scenarios:

1. **Student Benefits**: Classic card + student context
2. **Traveler Benefits**: Infinite card + traveler context  
3. **Tamil Translation**: Any request with `"preferred_language": "ta"`
4. **Error Handling**: Try unmasked card to see validation

---

**Ready to start?** Run: `cd backend && python main.py`


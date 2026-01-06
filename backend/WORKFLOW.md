# Visa Benefits Workflow - Complete Execution Flow

## Overview

This document describes the step-by-step execution workflow for the Visa Benefits system, from user input to final compliant response. The system uses 7 specialized agents orchestrated via LangGraph, with all processing done using local models (Ollama + HuggingFace).

---

## 1. INPUT FORMAT

### User Request Structure
```json
{
  "card_number": "4111-****-****-1111",
  "user_context": "student",  // Optional: student | traveler | family
  "preferred_language": "en",  // "en" or "ta"
  "location": "Chennai"  // Optional: city-level only
}
```

### Validation Rules
- **Card Number**: MUST match pattern `4[0-9]{3}-\*{4}-\*{4}-[0-9]{4}`
- **Rejection**: Full PANs (unmasked) are immediately rejected
- **Context**: Must be one of: student, traveler, family (if provided)
- **Language**: Must be "en" or "ta"

---

## 2. SEQUENTIAL EXECUTION FLOW

### Step 1: Supervisor Agent (Entry Point)
**Trigger**: API request received  
**Active Agent**: Supervisor/Orchestrator  
**Tools**: None (state management only)  
**Input**: Raw user request  
**Output**: Validated state passed to Card Intelligence  
**Validation**: None (pass-through)

**Data Contract**:
```python
{
  "card_number": str,
  "user_context": Optional[str],
  "preferred_language": "en" | "ta",
  "location": Optional[str]
}
```

---

### Step 2: Card Intelligence Agent
**Trigger**: After Supervisor  
**Active Agent**: Card Intelligence  
**Tools**: Regex pattern matching, BIN lookup table  
**Input**: Card number string  
**Output**: Tier detection + validation status  
**Validation**: 
- Pattern match against masked Visa format
- Reject full PANs
- BIN lookup for tier determination

**Decision Points**:
- ✅ Valid masked card → Continue to Retrieval
- ❌ Invalid format → Route to Compliance (error path)
- ❌ Unsupported BIN → Route to Compliance (error path)

**Data Contract**:
```python
{
  "detected_tier": "Classic" | "Signature" | "Infinite",
  "bin_valid": bool,
  "error": Optional[str],
  "error_code": Optional[str]  // "INVALID_CARD_FORMAT" | "UNSUPPORTED_BIN"
}
```

**BIN Mapping**:
- `4000`, `4333` → Signature
- `4111` → Classic
- `4222`, `4444` → Infinite

---

### Step 3: Benefit Retrieval Agent
**Trigger**: After Card Intelligence (if valid)  
**Active Agent**: Benefit Retrieval  
**Tools**: 
- HuggingFace Embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- ChromaDB vector database
- Semantic similarity search

**Input**: 
- Card tier
- User context
- Location (optional)

**Output**: Structured benefit chunks with metadata  
**Validation**: 
- Ensure at least one benefit found
- Verify RAG grounding (all benefits from documents)

**Decision Points**:
- ✅ Benefits found → Continue to Explanation
- ❌ No benefits → Route to Compliance (error path)

**Data Contract**:
```python
{
  "retrieved_docs": List[Document],
  "benefit_chunks": [
    {
      "content": str,
      "metadata": dict,
      "similarity_score": float,
      "source_chunk_id": str
    }
  ],
  "error": Optional[str],
  "error_code": Optional[str]  // "NO_BENEFITS_FOUND" | "RETRIEVAL_ERROR"
}
```

**RAG Query Construction**:
```
Query = "Visa {tier} {context} {location}"
Example: "Visa Signature student Chennai"
```

**Retrieval Parameters**:
- Top K: 5 chunks
- Similarity threshold: None (returns all top K)
- Embedding model: `all-MiniLM-L6-v2` (384 dimensions)

---

### Step 4: Explanation Agent
**Trigger**: After Benefit Retrieval (if benefits found)  
**Active Agent**: Explanation  
**Tools**: 
- ChatOllama (model: `llama3.2`)
- Prompt template for plain language conversion

**Input**: 
- Official benefit text (from RAG)
- Card tier
- User context

**Output**: Plain language explanation  
**Validation**: 
- Minimum length check (prevent hallucinations)
- Source verification (only use retrieved chunks)

**Decision Points**:
- ✅ Valid explanation → Continue to Recommendation
- ❌ Too short/invalid → Route to Compliance (error path)

**Data Contract**:
```python
{
  "plain_language_explanation": str,  // 2-3 sentences, actionable
  "error": Optional[str],
  "error_code": Optional[str]  // "EXPLANATION_ERROR"
}
```

**Prompt Template**:
```
You are a Visa Benefit Expert. Your task is to explain benefits in plain, actionable language.

CARD TIER: {tier}
USER CONTEXT: {context}

OFFICIAL BENEFIT RULES (DO NOT INVENT ANYTHING):
{official_text}

TASK:
1. Identify the most relevant benefit for this user
2. Explain it in 2-3 clear, simple sentences
3. Include specific details (amounts, dates, requirements) from the official rules
4. Make it actionable (what the user needs to do)
5. DO NOT add any benefits not mentioned in the official rules above
```

**LLM Parameters**:
- Model: `llama3.2`
- Temperature: 0.3 (balanced creativity/consistency)
- Max tokens: 200

---

### Step 5: Recommendation Agent
**Trigger**: After Explanation  
**Active Agent**: Recommendation  
**Tools**: Weighted scoring algorithm  
**Input**: 
- Benefit chunks
- User context
- Location

**Output**: Ranked benefits with scores  
**Validation**: None (deterministic scoring)

**Decision Points**: Always continues (no errors possible)

**Data Contract**:
```python
{
  "ranked_benefits": [
    {
      "chunk": dict,
      "scores": {
        "lifestyle": float,  // 0-1
        "location": float,   // 0-1
        "temporal": float,   // 0-1
        "monetary": float,   // 0-1
        "total": float       // weighted sum
      }
    }
  ],
  "top_benefit": dict  // Highest scoring benefit
}
```

**Scoring Weights**:
- Lifestyle Relevance: 35%
- Location Proximity: 20%
- Temporal Applicability: 25%
- Monetary Value: 20%

**Scoring Logic**:
1. **Lifestyle**: Keyword matching against context
   - student → education, campus, university
   - traveler → travel, lounge, airport, flight
   - family → family, dining, shopping, protection

2. **Location**: 
   - Exact match in content → 1.0
   - "global"/"worldwide" → 0.8
   - Default → 0.5

3. **Temporal**:
   - Valid until 2026/2027 → 1.0
   - Valid until 2025 → 0.7
   - Expired/ended → 0.1
   - Default → 0.5

4. **Monetary**:
   - Extract `$X` amounts from content
   - $500+ → 1.0
   - $100-$499 → 0.7
   - $50-$99 → 0.5
   - <$50 → 0.3

**No Transaction History**: This agent uses NO user transaction data, only static benefit content and user-provided context.

---

### Step 6: Language Agent
**Trigger**: After Recommendation  
**Active Agent**: Language  
**Tools**: 
- ChatOllama (model: `llama3.2`)
- Translation prompt template

**Input**: 
- Plain language explanation
- Target language (en/ta)

**Output**: Translated response  
**Validation**: 
- Translation length check (fallback if too short)
- Error handling (fallback to English)

**Decision Points**:
- ✅ Valid translation → Continue to Compliance
- ⚠️ Translation error → Use English fallback, continue to Compliance

**Data Contract**:
```python
{
  "translated_response": str,  // Tamil or English
  "error": Optional[str]  // Non-fatal: translation fallback
}
```

**Translation Prompt**:
```
Translate the following Visa benefit explanation to Tamil (தமிழ்).
Preserve all financial terms, numbers, and card details exactly.
Keep the tone professional and clear.

ENGLISH TEXT:
{text}

TAMIL TRANSLATION:
```

**LLM Parameters**:
- Model: `llama3.2`
- Temperature: 0.1 (low for accuracy)
- Fallback: Return English if translation fails

---

### Step 7: Compliance Agent (Final Veto)
**Trigger**: After Language (or from any error path)  
**Active Agent**: Compliance  
**Tools**: 
- Regex pattern matching (unsafe language detection)
- Disclaimer injection
- PCI/GDPR/CCPA checks

**Input**: 
- Translated response (or error state)
- All previous agent outputs

**Output**: Final compliant response  
**Validation**: 
- Unsafe language detection
- PAN detection (reject if found)
- Disclaimer injection

**Decision Points**:
- ✅ Compliant → Return success response
- ❌ Non-compliant → Return error response

**Data Contract**:
```python
{
  "final_output": {
    "status": "success" | "error",
    "card_tier": Optional[str],
    "recommended_benefit": {
      "explanation": str,
      "source_chunks": [
        {
          "chunk_id": str,
          "similarity": float
        }
      ]
    },
    "disclaimers": List[str],
    "language": "en" | "ta",
    "metadata": {
      "bin_validated": bool,
      "rag_grounded": bool,
      "compliance_approved": bool
    },
    "error_code": Optional[str],
    "message": Optional[str]
  },
  "compliance_approved": bool,
  "disclaimers": List[str]
}
```

**Compliance Checks**:
1. **Unsafe Language**: Detect and replace
   - Patterns: "guaranteed", "always", "never", "100%", "definitely"
   - Replacement: "may" (softer language)

2. **PAN Detection**: 
   - Pattern: `\d{4}-\d{4}-\d{4}-\d{4}`
   - Action: Reject immediately (compliance violation)

3. **Required Disclaimers**:
   - "Terms and conditions apply. Benefits subject to cardholder agreement."
   - "This is an awareness-only system. No transactions or account actions are performed."
   - "Generated locally. No card data is stored or logged."

4. **Metadata Tracking**:
   - BIN validation status
   - RAG grounding confirmation
   - Compliance approval status

---

## 3. ERROR HANDLING & EDGE CASES

### Error Paths

#### A. Invalid Card Format
**Trigger**: Card Intelligence Agent  
**Error Code**: `INVALID_CARD_FORMAT`  
**Flow**: Card Intel → Compliance (error) → END  
**Response**:
```json
{
  "status": "error",
  "error_code": "INVALID_CARD_FORMAT",
  "message": "Invalid card format. Only masked Visa cards accepted..."
}
```

#### B. Unsupported BIN
**Trigger**: Card Intelligence Agent  
**Error Code**: `UNSUPPORTED_BIN`  
**Flow**: Card Intel → Compliance (error) → END  
**Response**:
```json
{
  "status": "error",
  "error_code": "UNSUPPORTED_BIN",
  "message": "Unsupported BIN: XXXX. Card tier cannot be determined."
}
```

#### C. No Benefits Found
**Trigger**: Benefit Retrieval Agent  
**Error Code**: `NO_BENEFITS_FOUND`  
**Flow**: Retrieval → Compliance (error) → END  
**Response**:
```json
{
  "status": "error",
  "error_code": "NO_BENEFITS_FOUND",
  "message": "No benefits found for this card tier and context."
}
```

#### D. Retrieval Error
**Trigger**: Benefit Retrieval Agent (exception)  
**Error Code**: `RETRIEVAL_ERROR`  
**Flow**: Retrieval → Compliance (error) → END  
**Response**:
```json
{
  "status": "error",
  "error_code": "RETRIEVAL_ERROR",
  "message": "Retrieval error: {exception_message}"
}
```

#### E. Explanation Error
**Trigger**: Explanation Agent  
**Error Code**: `EXPLANATION_ERROR`  
**Flow**: Explanation → Compliance (error) → END  
**Response**:
```json
{
  "status": "error",
  "error_code": "EXPLANATION_ERROR",
  "message": "Explanation generation error: {exception_message}"
}
```

#### F. Language Translation Fallback
**Trigger**: Language Agent  
**Error Code**: None (non-fatal)  
**Flow**: Language → Compliance (with English fallback) → END  
**Behavior**: Uses English if Tamil translation fails or is too short

#### G. Compliance Rejection
**Trigger**: Compliance Agent  
**Error Code**: `COMPLIANCE_REJECTED`  
**Flow**: Compliance → END  
**Response**:
```json
{
  "status": "error",
  "error_code": "COMPLIANCE_REJECTED",
  "message": "Response blocked for security compliance."
}
```

---

## 4. DATA FLOW DIAGRAM

```
User Input
    ↓
[Supervisor] → State initialization
    ↓
[Card Intelligence] → Validate card, detect tier
    ├─ Valid → [Benefit Retrieval]
    └─ Invalid → [Compliance] (error) → END
    ↓
[Benefit Retrieval] → RAG search
    ├─ Found → [Explanation]
    └─ Not Found → [Compliance] (error) → END
    ↓
[Explanation] → Plain language conversion
    ↓
[Recommendation] → Rank benefits by score
    ↓
[Language] → Translate if needed
    ├─ Success → [Compliance]
    └─ Error → [Compliance] (English fallback)
    ↓
[Compliance] → Final validation, disclaimers
    ├─ Approved → Success Response
    └─ Rejected → Error Response
    ↓
END
```

---

## 5. FINAL RESPONSE ASSEMBLY

### Success Response Structure
```json
{
  "status": "success",
  "card_tier": "Signature",
  "recommended_benefit": {
    "explanation": "As a Signature cardholder, you get 1 free lounge access per quarter...",
    "source_chunks": [
      {
        "chunk_id": "chunk_001",
        "similarity": 0.87
      }
    ]
  },
  "disclaimers": [
    "Terms and conditions apply. Benefits subject to cardholder agreement.",
    "This is an awareness-only system. No transactions or account actions are performed.",
    "Generated locally. No card data is stored or logged."
  ],
  "language": "en",
  "metadata": {
    "bin_validated": true,
    "rag_grounded": true,
    "compliance_approved": true
  }
}
```

### Response Guarantees
1. ✅ **RAG-Grounded**: Every benefit maps to a source document chunk
2. ✅ **Privacy-Compliant**: No card data stored or logged
3. ✅ **Awareness-Only**: No transaction capabilities
4. ✅ **Compliance-Approved**: All disclaimers included
5. ✅ **Language-Localized**: Translated to user's preference (if supported)

---

## 6. IMPLEMENTATION NOTES

### Local Model Requirements
- **LLM**: Ollama with `llama3.2` model
- **Embeddings**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB (local persistence)

### Setup Steps
1. Install Ollama and pull `llama3.2`:
   ```bash
   ollama pull llama3.2
   ```

2. Ingest benefits data:
   ```bash
   python backend/ingest.py
   ```

3. Start FastAPI server:
   ```bash
   python backend/main.py
   ```

### Testing
```bash
curl -X POST http://localhost:8000/benefits \
  -H "Content-Type: application/json" \
  -d '{
    "card_number": "4111-****-****-1111",
    "user_context": "student",
    "preferred_language": "en",
    "location": "Chennai"
  }'
```

---

## 7. CONSTRAINTS VERIFICATION

✅ **Masked Cards Only**: Regex validation in Card Intelligence Agent  
✅ **No Data Persistence**: All processing in-memory, no databases for sessions  
✅ **Awareness-Only**: No transaction endpoints, explicit disclaimers  
✅ **RAG-Grounded**: All benefits from vector database, no hallucinations  
✅ **Agent Specialization**: Each agent has single responsibility  
✅ **Compliance Veto**: Final agent can reject non-compliant output  

---

## 8. HACKATHON DEMO SCENARIOS

### Scenario 1: Student with Classic Card
- Input: `4111-****-****-1111`, context: `student`, location: `Chennai`
- Expected: Educational benefits, student discounts
- Language: English

### Scenario 2: Traveler with Infinite Card
- Input: `4222-****-****-2222`, context: `traveler`, location: `Mumbai`
- Expected: Lounge access, concierge services
- Language: Tamil

### Scenario 3: Invalid Card Format
- Input: `4111-1234-5678-1111` (unmasked)
- Expected: Error response with `INVALID_CARD_FORMAT`

### Scenario 4: No Benefits Found
- Input: Unsupported tier/context combination
- Expected: Error response with `NO_BENEFITS_FOUND`

---

**END OF WORKFLOW DOCUMENTATION**


# Visa Benefits System - Architecture

## System Overview

A **truly autonomous agentic AI system** for Visa card benefits recommendation built with:
- **Backend**: FastAPI + LangGraph (7-agent orchestration)
- **Frontend**: Next.js React application
- **LLM**: Ollama (llama3.2) for text generation and reasoning
- **Vector DB**: ChromaDB with HuggingFace embeddings
- **Data Source**: `data/benefits.txt` (structured text file)

---

## How Generative AI Works

### Purpose
Generative AI converts structured/technical data into natural, human-readable language.

### Where It's Used

#### 1. **Explanation Agent** (Lines 355-453)
- **Input**: Retrieved benefit chunks (legal/technical text)
- **Process**: Uses Ollama LLM (llama3.2) with temperature=0.3
- **Output**: Plain-language explanations in 2-3 sentences
- **RAG-Grounded**: Only uses information from retrieved documents
- **Example**: Converts "15% off on printing services" → "You can save 15% on printing and binding services at participating locations..."

#### 2. **Language Agent** (Lines 633-708)
- **Input**: English explanation
- **Process**: Uses Ollama LLM (llama3.2) with temperature=0.1 (low for translation)
- **Output**: Tamil translation preserving financial terms and numbers
- **Fallback**: Returns English if translation fails

### Key Characteristics
- ✅ **RAG-Grounded**: All generation based on retrieved documents
- ✅ **No Hallucination**: Prompt explicitly forbids inventing facts
- ✅ **Local Processing**: No external API calls, runs on Ollama
- ✅ **Temperature Control**: Low temperature (0.1-0.3) for accuracy

---

## How Agentic AI Works

### What Makes It Agentic

The system is **truly autonomous** with these capabilities:

#### 1. **Tool Use** 🛠️
Agents autonomously call tools from a registry:
- `search_benefits`: Semantic search in vector database
- `validate_card`: Card validation and tier detection
- `filter_by_location`: Location-based filtering
- `score_benefits`: Benefit ranking and scoring
- `check_compliance`: Compliance violation detection

**Location**: `backend/app/tools.py`

#### 2. **Autonomous Decision-Making** 🧠
Agents use **LLM reasoning** to make decisions:
- **Supervisor Agent**: Creates goals and plans using LLM
- **Card Intelligence Agent**: Decides whether to reject or continue with default tier
- **Benefit Retrieval Agent**: Uses LLM to build optimal search queries
- **Compliance Agent**: Decides whether to fix, reject, or proceed with violations

**Example**:
```python
# Card Intelligence Agent uses LLM to decide:
decision = llm.invoke("Should we reject or continue?")
if "continue" in decision:
    next_action = "retrieval"  # Proceed with default tier
else:
    next_action = "compliance"  # Reject request
```

#### 3. **Autonomous Routing** 🎯
LLM-powered router decides next actions:
- Considers: errors, prerequisites, goal progress, agent reasoning
- Can skip steps (e.g., skip translation if already in target language)
- Can take alternative paths based on context
- Can end workflow early if goal achieved

**Function**: `autonomous_router()` in `graph.py`

#### 4. **Iterative Planning** 📋
Agents can **replan** if results are insufficient:
- **Benefit Retrieval Agent**: Expands search if too few results found
- **Supervisor Agent**: Can revise plan based on progress
- Plan iteration counter tracks replanning attempts

#### 5. **Self-Correction** 🔄
Agents can **retry and adjust** their approach:
- Each agent has a retry counter (max 2 retries)
- Agents can retry tool calls on failure
- Agents adjust strategy based on errors

#### 6. **Memory System** 💾
System maintains **conversation history and agent memory**:
- `conversation_history`: Tracks agent actions and decisions
- `agent_memory`: Stores agent-specific learnings
- `tool_calls`: History of all tool invocations
- `agent_reasoning`: LLM reasoning from each agent

#### 7. **Goal-Oriented Behavior** 🎯
Agents work towards **goals with sub-goals**:
- **Supervisor Agent** creates main goal and sub-goals using LLM
- Each agent works towards specific sub-goals
- Progress tracked through goal completion

### Agent Workflow

```
Supervisor → Card Intelligence → Benefit Retrieval → Explanation 
    → Recommendation → Language → Compliance → End
```

**Key**: Each agent can autonomously decide to:
- Skip steps
- Take alternative paths
- Retry on errors
- End early if goal achieved

### State Management

```python
class AgentState(TypedDict):
    # Agentic AI fields
    goal: Optional[str]  # Main goal
    sub_goals: List[str]  # Sub-goals
    plan: Optional[str]  # Current plan
    conversation_history: List[Dict]  # Past interactions
    agent_memory: Dict[str, Any]  # Agent learnings
    tool_calls: List[Dict]  # Tool call history
    agent_reasoning: Dict[str, str]  # LLM reasoning
    next_action: Optional[str]  # Agent-decided next step
    retry_count: Dict[str, int]  # Retry counts
    # ... standard fields
```

---

## RAG Implementation

### How RAG Works

1. **Retrieval**: Semantic search using vector embeddings
   - Query: `"Visa {tier} {context} {location}"`
   - Vector similarity search in ChromaDB
   - Returns top 20-30 chunks with similarity scores

2. **Augmentation**: Query includes user context
   - Card tier, user context (student/traveler), location
   - Location variants for better matching

3. **Generation**: LLM generates explanation from retrieved chunks
   - **RAG-Grounded**: Only uses information from retrieved documents
   - Prompt explicitly forbids inventing facts
   - All explanations must cite source chunks

### Where RAG is Used

- **Benefit Retrieval Agent**: Performs semantic search
- **Explanation Agent**: Generates explanations from retrieved chunks
- **Language Agent**: Translates RAG-grounded explanations

---

## Critical Questions Answered

### 1. What Happens If Benefits Change?

**Answer**: Manual re-ingestion required.

- Update `data/benefits.txt`
- Run `python ingest_simple.py`
- Old DB deleted, new one created
- System uses new data immediately

**Limitations**:
- No automatic updates
- No versioning
- Must re-process entire file
- Requires developer intervention

**Better Approach**: Incremental updates, version control, API endpoint for re-ingestion

---

### 2. How Do You Avoid Wrong Recommendations?

**Answer**: Multiple safeguards in place.

1. **RAG Grounding**: All explanations come from retrieved documents
2. **Location Filtering**: Strict matching (rejects cross-city benefits)
3. **Tier Matching**: Only retrieves benefits for detected card tier
4. **Compliance Agent**: Final check for unsafe language and PAN detection
5. **Scoring System**: Weighted scoring (lifestyle, location, temporal, monetary)

**Weaknesses**:
- No source verification after generation
- No confidence thresholds
- No user feedback loop
- Relies on prompt engineering

**Recommendations**: Add similarity score thresholds, post-generation fact-checking, benefit expiry validation

---

### 3. Why Not Use Transaction Data?

**Answer**: Privacy and compliance design decision.

**Why Not Used**:
1. **PCI Compliance**: Transaction data requires PCI DSS Level 1 compliance
2. **Privacy Regulations**: GDPR/CCPA compliance, data minimization
3. **System Purpose**: "Awareness-only" system, no transactions
4. **Architecture Simplicity**: Stateless design, no user data storage

**Trade-offs**:
- ✅ Pros: Privacy, compliance, simplicity
- ❌ Cons: Less personalized, no usage tracking, generic recommendations

---

### 4. How Is This Compliant with PCI?

**Answer**: System is likely **OUTSIDE PCI scope**.

**Compliant Aspects**:
1. **No Full PAN Storage**: Only masked cards accepted (format: `4XXX-****-****-XXXX`)
2. **No Data Persistence**: Stateless design, no user data stored
3. **Awareness-Only**: No transactions performed, read-only information
4. **Compliance Agent**: PAN detection in output, prevents leakage

**Potential Gaps** (for production):
- BIN data should be encrypted
- Ensure no card data in error logs
- Add TLS/HTTPS enforcement
- Restrict CORS origins
- Add rate limiting

---

### 5. What Part Is Hard-Coded?

**Answer**: Several components are hard-coded.

1. **BIN to Tier Mapping**: Only 5 BIN ranges (mock data)
2. **Location Mapping**: Only 4 cities
3. **Scoring Weights**: Fixed weights (lifestyle: 0.35, location: 0.30, etc.)
4. **Category Keywords**: Hard-coded keyword lists
5. **Unsafe Language Patterns**: Rule-based pattern list
6. **LLM Model Name**: "llama3.2" hard-coded
7. **Embedding Model**: "all-MiniLM-L6-v2" hard-coded
8. **Retrieval Count**: Fixed to 20 results
9. **Recommendation Count**: Always returns top 4
10. **Disclaimers**: Fixed text, not localized

**Recommendation**: Move to config file, use environment variables, database for BIN mapping

---

### 6. What Fails at Scale?

**Answer**: Several scalability bottlenecks.

1. **Vector Database**: Local SQLite, fails at >100K concurrent queries
   - **Solution**: Migrate to distributed vector DB (Pinecone, Weaviate, Qdrant)

2. **LLM Processing**: Local Ollama, fails at >10 concurrent requests
   - **Solution**: Async processing, load balancing, or cloud LLM API

3. **Embedding Generation**: CPU-bound, fails at >100K benefit entries
   - **Solution**: Batch processing, GPU acceleration

4. **State Management**: In-memory, no caching
   - **Solution**: Redis for state caching, session management

5. **Location Filtering**: O(n) string matching, fails at >1000 chunks
   - **Solution**: Pre-filter in vector query, metadata filtering

6. **API Endpoint**: Synchronous, no rate limiting, fails at >1000 RPS
   - **Solution**: Async endpoints, rate limiting, response caching

**Performance Estimates**:
- Vector Search: ~100 QPS
- LLM Generation: ~5 concurrent
- API Endpoint: ~50 RPS

---

## System Strengths

- ✅ Privacy-first design (no data persistence)
- ✅ RAG-grounded responses (reduces hallucinations)
- ✅ Truly autonomous agentic AI (LLM reasoning, tool use, planning)
- ✅ Local processing (no external API dependencies)
- ✅ Compliance-aware (masked cards, no PAN storage)

## System Weaknesses

- ❌ Limited scalability (local components)
- ❌ Hard-coded configurations
- ❌ No transaction data (less personalized)
- ❌ Manual benefit updates
- ❌ Basic error handling

---

## Summary

This is a **truly autonomous agentic AI system** that:
- Uses **Generative AI** for natural language generation (explanations, translations)
- Uses **Agentic AI** for autonomous decision-making, tool use, planning, and self-correction
- Uses **RAG** to ground all responses in source documents
- Maintains **memory** and **goal-oriented behavior**
- Makes **intelligent routing decisions** based on LLM reasoning

The system is production-ready for small-scale use but requires scalability improvements for large-scale deployment.


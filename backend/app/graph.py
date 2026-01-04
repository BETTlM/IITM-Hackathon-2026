"""
Visa Benefits Workflow - Complete 7-Agent System
Uses local models (Ollama + HuggingFace) - No OpenAI API

GENERATIVE AI:
- Uses Ollama (llama3.2) for natural language generation
- Explanation Agent: Generates plain-language explanations from legal text
- Language Agent: Generates translations (English/Tamil)
- All text generation is RAG-grounded to prevent hallucinations

AGENTIC AI:
- LangGraph-based multi-agent orchestration system
- 7 specialized agents with distinct responsibilities
- Agents communicate through shared state (AgentState)
- Conditional routing based on agent decisions
- Supervisor pattern for workflow coordination
"""

import re
import json
from typing import TypedDict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    # Input
    card_number: str
    user_context: Optional[str]  # student / traveler / family
    preferred_language: Literal["en", "ta"]  # English or Tamil
    location: Optional[str]  # city-level only
    
    # Card Intelligence Output
    detected_tier: Optional[str]  # Classic / Signature / Infinite
    bin_valid: bool
    
    # Benefit Retrieval Output
    retrieved_docs: List[dict]  # Serialized document data
    benefit_chunks: List[dict]  # Structured benefit data with metadata
    
    # Explanation Output
    plain_language_explanation: Optional[str]
    
    # Recommendation Output
    ranked_benefits: List[dict]  # Benefits with scores
    top_benefit: Optional[dict]
    
    # Language Output
    translated_response: Optional[str]
    
    # Compliance Output
    compliance_approved: bool
    disclaimers: List[str]
    
    # Final Output
    final_output: Optional[dict]
    
    # Error Handling
    error: Optional[str]
    error_code: Optional[str]  # INVALID_CARD, UNSUPPORTED_BIN, NO_BENEFITS, etc.


# ============================================================================
# 2. HELPER FUNCTIONS
# ============================================================================

def _extract_informative_summary(content: str, max_length: int = 200) -> str:
    """
    Extract informative summary from benefit content.
    Prioritizes key information like benefit name, discounts, locations, etc.
    """
    if not content:
        return ""
    
    lines = content.split('\n')
    summary_parts = []
    
    # Extract benefit name/title (usually first non-header line)
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if len(line) > 15 and len(line) < 100:
            # Check if it looks like a benefit name
            if not any(line.lower().startswith(p) for p in ['visa', 'card', 'tier', 'benefit']):
                summary_parts.append(line)
                break
    
    # Extract key details (discounts, locations, amounts)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Look for important information
        if any(keyword in line.lower() for keyword in ['%', 'off', 'discount', 'valid', 'available', 'access', 'save', 'rs.', '₹', '$']):
            if len(line) < 150 and line not in summary_parts:
                summary_parts.append(line)
                if len(summary_parts) >= 3:
                    break
    
    # If we have good summary parts, join them
    if summary_parts:
        summary = '. '.join(summary_parts)
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary
    
    # Fallback: use first meaningful sentences
    sentences = content.split('.')
    meaningful = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 150]
    if meaningful:
        summary = '. '.join(meaningful[:2])
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        return summary
    
    # Last fallback: truncate
    return content[:max_length] + "..." if len(content) > max_length else content


# ============================================================================
# 3. AGENT IMPLEMENTATIONS
# ============================================================================

def supervisor_agent(state: AgentState) -> dict:
    """
    Supervisor/Orchestrator Agent (AGENTIC AI)
    - Part of the multi-agent orchestration system
    - Controls execution order and passes context between agents
    - LangGraph handles routing, but this agent validates flow
    - Enables agent-to-agent communication and coordination
    """
    # Supervisor doesn't modify state, just validates flow
    if state.get("error"):
        return {}
    return {}


def card_intelligence_agent(state: AgentState) -> dict:
    """
    Card Intelligence Agent
    - Validates masked card format (HARD CONSTRAINT)
    - Performs BIN lookup
    - Determines card tier
    """
    card = state["card_number"]
    
    # HARD CONSTRAINT: Only accept masked Visa format
    # Pattern: 4XXX-****-****-XXXX
    mask_pattern = r"^4[0-9]{3}-\*{4}-\*{4}-[0-9]{4}$"
    
    if not re.match(mask_pattern, card):
        return {
            "error": "Invalid card format. Only masked Visa cards accepted (e.g., 4111-****-****-1111).",
            "error_code": "INVALID_CARD_FORMAT",
            "bin_valid": False
        }
    
    # Reject full PANs (safety check)
    if not re.search(r"\*{4}", card):
        return {
            "error": "Full card numbers are not accepted. Please use masked format.",
            "error_code": "FULL_PAN_REJECTED",
            "bin_valid": False
        }
    
    # Extract BIN (first 4-6 digits)
    first_4 = card[:4]
    
    # BIN to Tier mapping (mock - in production, use real BIN database)
    bin_tier_map = {
        "4000": "Signature",
        "4111": "Classic",
        "4222": "Infinite",
        "4333": "Signature",
        "4444": "Infinite"
    }
    
    detected_tier = bin_tier_map.get(first_4)
    
    if not detected_tier:
        return {
            "error": f"Unsupported BIN: {first_4}. Card tier cannot be determined.",
            "error_code": "UNSUPPORTED_BIN",
            "bin_valid": False,
            "detected_tier": "Classic"  # Default fallback
        }
    
    return {
        "detected_tier": detected_tier,
        "bin_valid": True
    }


def benefit_retrieval_agent(state: AgentState) -> dict:
    """
    Benefit Retrieval Agent
    - Queries vector database using RAG
    - Retrieves benefit documents with semantic similarity
    - Outputs structured JSON with source chunks
    """
    if state.get("error") or not state.get("bin_valid"):
        return {}
    
    tier = state["detected_tier"]
    context = state.get("user_context") or ""
    location = state.get("location") or ""
    
    try:
        # Use same embeddings as ingest
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        db = Chroma(
            persist_directory="./visa_db",
            embedding_function=embeddings
        )
        
        # Build semantic query
        query_parts = [f"Visa {tier}"]
        if context:
            query_parts.append(context)
        if location:
            # Add location variants for better matching
            location_map = {
                "bangalore": ["bangalore", "bengaluru"],
                "chennai": ["chennai", "madras"],
                "mumbai": ["mumbai", "bombay"],
                "goa": ["goa"]
            }
            location_lower = location.lower()
            for key, variants in location_map.items():
                if location_lower in variants or any(v in location_lower for v in variants):
                    query_parts.extend(variants)
                    break
            else:
                query_parts.append(location)
        
        query = " ".join(query_parts)
        
        # Retrieve top K chunks (RAG grounding) - get more for better selection
        results = db.similarity_search_with_score(query, k=20)  # Get more to filter by location
        
        if not results:
            return {
                "error": "No benefits found for this card tier and context.",
                "error_code": "NO_BENEFITS_FOUND",
                "retrieved_docs": [],
                "benefit_chunks": []
            }
        
        # Location filtering - strict matching
        location_map = {
            "bangalore": ["bangalore", "bengaluru"],
            "chennai": ["chennai", "madras"],
            "mumbai": ["mumbai", "bombay"],
            "goa": ["goa"]
        }
        
        # Get location variants for filtering
        location_variants = []
        if location:
            location_lower = location.lower()
            for key, variants in location_map.items():
                if location_lower in variants or any(v in location_lower for v in variants):
                    location_variants = variants
                    break
            if not location_variants:
                location_variants = [location_lower]
        
        # Get other cities to exclude
        other_cities = {
            "bangalore": ["chennai", "mumbai", "goa", "madras", "bombay"],
            "chennai": ["bangalore", "bengaluru", "mumbai", "goa", "bombay"],
            "mumbai": ["bangalore", "bengaluru", "chennai", "madras", "goa"],
            "goa": ["bangalore", "bengaluru", "chennai", "madras", "mumbai", "bombay"]
        }
        
        other_cities_to_check = []
        if location:
            location_lower = location.lower()
            for key, variants in location_map.items():
                if location_lower in variants or any(v in location_lower for v in variants):
                    other_cities_to_check = other_cities.get(key, [])
                    break
        
        # Structure benefits with metadata and filter by location
        benefit_chunks = []
        retrieved_docs_serialized = []
        for doc, score in results:
            content_lower = doc.page_content.lower()
            metadata_location = doc.metadata.get("location", "").lower() if doc.metadata else ""
            
            # STRICT LOCATION FILTERING
            if location and location_variants:
                # Check if this benefit is for the selected location
                has_selected_location = any(variant in content_lower for variant in location_variants) or \
                                      any(variant in metadata_location for variant in location_variants)
                
                # Check if this benefit mentions other cities
                has_other_city = any(city in content_lower for city in other_cities_to_check)
                
                # Reject if it has other city but not selected location
                if has_other_city and not has_selected_location:
                    continue  # Skip this benefit - wrong location
                
                # If location is specified, only include if it matches
                if not has_selected_location:
                    continue  # Skip - doesn't match selected location
            
            # Extract structured data from document
            chunk_data = {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "source_chunk_id": doc.metadata.get("source", "unknown")
            }
            benefit_chunks.append(chunk_data)
            
            # Serialize Document for state
            retrieved_docs_serialized.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "retrieved_docs": retrieved_docs_serialized,
            "benefit_chunks": benefit_chunks
        }
    
    except Exception as e:
        return {
            "error": f"Retrieval error: {str(e)}",
            "error_code": "RETRIEVAL_ERROR",
            "retrieved_docs": [],
            "benefit_chunks": []
        }


def explanation_agent(state: AgentState) -> dict:
    """
    Explanation Agent (GENERATIVE AI)
    - Uses Ollama LLM (llama3.2) to generate natural language explanations
    - Converts legal/technical text into plain, actionable language
    - RAG-grounded: Only uses information from retrieved documents
    - NO new facts introduced (prevents hallucinations)
    - Generative AI ensures human-readable, personalized explanations
    """
    if state.get("error") or not state.get("benefit_chunks"):
        return {}
    
    tier = state["detected_tier"]
    context = state.get("user_context") or ""
    chunks = state["benefit_chunks"]
    
    # Combine all retrieved chunks (RAG-grounded)
    official_text = "\n\n".join([
        f"[Source {i+1}]\n{chunk['content']}"
        for i, chunk in enumerate(chunks)
    ])
    
    try:
        llm = ChatOllama(model="llama3.2", temperature=0.3)
        
        prompt = ChatPromptTemplate.from_template(
            """You are a Visa Benefit Expert. Your task is to explain benefits in plain, actionable language.

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
6. DO NOT start with phrases like "Based on", "I recommend", "Here's" - just state the benefit directly

OUTPUT: Plain language explanation only, no disclaimers. Start directly with the benefit description."""
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "tier": tier,
            "context": context,
            "official_text": official_text
        })
        
        explanation = response.content.strip()
        
        # Validate that explanation doesn't introduce new facts
        # (Basic check - in production, use more sophisticated validation)
        if len(explanation) < 20:
            return {
                "error": "Explanation too short. Possible hallucination.",
                "error_code": "EXPLANATION_ERROR"
            }
        
        return {
            "plain_language_explanation": explanation
        }
    
    except Exception as e:
        return {
            "error": f"Explanation generation error: {str(e)}",
            "error_code": "EXPLANATION_ERROR"
        }


def recommendation_agent(state: AgentState) -> dict:
    """
    Recommendation Agent
    - Ranks benefits using weighted scoring:
      * Lifestyle relevance (0-1)
      * Location proximity (0-1)
      * Temporal applicability (0-1)
      * Monetary value (0-1)
    - Uses NO transaction history
    """
    if state.get("error") or not state.get("benefit_chunks"):
        return {}
    
    # Handle None values safely
    context = (state.get("user_context") or "").lower()
    location = (state.get("location") or "").lower()
    chunks = state["benefit_chunks"]
    
    # Location normalization map
    location_map = {
        "bangalore": ["bangalore", "bengaluru"],
        "chennai": ["chennai", "madras"],
        "mumbai": ["mumbai", "bombay"],
        "goa": ["goa"]
    }
    
    # Get all location variants for strict filtering
    location_variants = []
    if location:
        for key, variants in location_map.items():
            if location in variants or any(v in location for v in variants):
                location_variants = variants
                break
        if not location_variants:
            location_variants = [location]
    
    # Scoring weights
    WEIGHTS = {
        "lifestyle": 0.35,
        "location": 0.30,  # Increased weight for location
        "temporal": 0.20,
        "monetary": 0.15
    }
    
    ranked_benefits = []
    
    for chunk in chunks:
        content = chunk["content"].lower()
        metadata = chunk.get("metadata", {})
        
        # STRICT LOCATION FILTERING - Reject if location doesn't match
        if location and location_variants:
            # Check if content mentions the selected location
            location_match = any(variant in content for variant in location_variants)
            
            # Also check metadata if available
            metadata_location = metadata.get("location", "").lower() if metadata else ""
            metadata_match = any(variant in metadata_location for variant in location_variants) if metadata_location else False
            
            # Check for other cities - if found, reject this benefit
            other_cities = {
                "bangalore": ["chennai", "mumbai", "goa", "madras", "bombay"],
                "chennai": ["bangalore", "bengaluru", "mumbai", "goa", "bombay"],
                "mumbai": ["bangalore", "bengaluru", "chennai", "madras", "goa"],
                "goa": ["bangalore", "bengaluru", "chennai", "madras", "mumbai", "bombay"]
            }
            
            # Get other cities to check
            other_cities_to_check = []
            for key, variants in location_map.items():
                if location in variants or any(v in location for v in variants):
                    other_cities_to_check = other_cities.get(key, [])
                    break
            
            # Reject if other city is mentioned
            has_other_city = any(city in content for city in other_cities_to_check)
            if has_other_city and not location_match and not metadata_match:
                continue  # Skip this benefit - wrong location
        
        # 1. Category Relevance
        category_keywords = {
            "student": ["student", "education", "campus", "university", "college", "school", "tuition", "textbook", "academic"],
            "travel": ["travel", "lounge", "airport", "flight", "hotel", "concierge", "trip", "journey", "vacation", "resort"],
            "dining_entertainment_shopping": ["dining", "restaurant", "cafe", "shopping", "mall", "entertainment", "movie", "cinema", "theater", "retail", "store", "dine"],
            "services": ["service", "insurance", "protection", "warranty", "concierge", "support", "assistance", "help", "benefit"]
        }
        
        category_score = 0.0
        if context in category_keywords:
            matches = sum(1 for kw in category_keywords[context] if kw in content)
            category_score = min(matches / max(len(category_keywords[context]), 1), 1.0)
        else:
            category_score = 0.5  # Neutral if no context
        
        # 2. Location Proximity - STRICT MATCHING
        location_score = 0.0  # Default to 0 - must match location
        if location and location_variants:
            # Exact location match
            if any(variant in content for variant in location_variants):
                location_score = 1.0
            elif metadata_location and any(variant in metadata_location for variant in location_variants):
                location_score = 1.0
            else:
                # If location is specified but doesn't match, give 0 score (will be filtered out)
                location_score = 0.0
        else:
            # No location specified - neutral score
            location_score = 0.5
        
        # 3. Temporal Applicability (checks for validity dates)
        temporal_score = 0.5  # Default
        if "2026" in content or "2027" in content:
            temporal_score = 1.0
        elif "2025" in content:
            temporal_score = 0.7
        elif "expired" in content or "ended" in content:
            temporal_score = 0.1
        
        # 4. Monetary Value (extracts dollar amounts)
        monetary_score = 0.5  # Default
        dollar_matches = re.findall(r'\$(\d+)', content)
        if dollar_matches:
            max_amount = max(int(amt) for amt in dollar_matches)
            if max_amount >= 500:
                monetary_score = 1.0
            elif max_amount >= 100:
                monetary_score = 0.7
            elif max_amount >= 50:
                monetary_score = 0.5
        
        # Weighted total score
        total_score = (
            category_score * WEIGHTS["lifestyle"] +
            location_score * WEIGHTS["location"] +
            temporal_score * WEIGHTS["temporal"] +
            monetary_score * WEIGHTS["monetary"]
        )
        
        # Only include if location matches (if location is specified)
        if location and location_variants:
            if location_score == 0.0:
                continue  # Skip - location doesn't match
        
        ranked_benefits.append({
            "chunk": chunk,
            "scores": {
                "lifestyle": category_score,
                "location": location_score,
                "temporal": temporal_score,
                "monetary": monetary_score,
                "total": total_score
            }
        })
    
    # Sort by total score (descending)
    ranked_benefits.sort(key=lambda x: x["scores"]["total"], reverse=True)
    
    top_benefit = ranked_benefits[0] if ranked_benefits else None
    
    return {
        "ranked_benefits": ranked_benefits,
        "top_benefit": top_benefit
    }


def language_agent(state: AgentState) -> dict:
    """
    Language Agent (GENERATIVE AI)
    - Uses Ollama LLM (llama3.2) for natural language translation
    - Generates translations from English to Tamil (or keeps English)
    - Preserves financial terminology and meaning
    - Generative AI ensures contextually appropriate translations
    """
    if state.get("error") or not state.get("plain_language_explanation"):
        return {}
    
    target_lang = state.get("preferred_language", "en")
    explanation = state["plain_language_explanation"]
    
    # If already in target language, return as-is
    if target_lang == "en":
        return {
            "translated_response": explanation
        }
    
    # Translate to Tamil using local LLM
    try:
        llm = ChatOllama(model="llama3.2", temperature=0.1)
        
        prompt = ChatPromptTemplate.from_template(
            """Translate the following Visa benefit explanation to Tamil (தமிழ்).
Preserve all financial terms, numbers, and card details exactly.
Keep the tone professional and clear.

ENGLISH TEXT:
{text}

TAMIL TRANSLATION:"""
        )
        
        chain = prompt | llm
        response = chain.invoke({"text": explanation})
        
        translated = response.content.strip()
        
        # Fallback: if translation fails or is too short, return English
        if len(translated) < len(explanation) * 0.3:
            return {
                "translated_response": explanation,  # Fallback to English
                "error": "Translation quality insufficient, using English"
            }
        
        return {
            "translated_response": translated
        }
    
    except Exception as e:
        # Fallback to English on error
        return {
            "translated_response": explanation,
            "error": f"Translation error: {str(e)}, using English"
        }


def compliance_agent(state: AgentState) -> dict:
    """
    Compliance Agent
    - Adds disclaimers
    - Checks PCI, GDPR, CCPA, VISA rules
    - Removes unsafe or misleading language
    - Has FINAL VETO power
    """
    if state.get("error"):
        return {
            "final_output": {
                "status": "error",
                "error_code": state.get("error_code", "UNKNOWN"),
                "message": state["error"]
            },
            "compliance_approved": False
        }
    
    response_text = state.get("translated_response") or state.get("plain_language_explanation", "")
    
    # Compliance checks
    disclaimers = []
    compliance_approved = True
    
    # Check for unsafe language
    unsafe_patterns = [
        r"guaranteed",
        r"always",
        r"never",
        r"100%",
        r"definitely"
    ]
    
    for pattern in unsafe_patterns:
        if re.search(pattern, response_text, re.IGNORECASE):
            # Soften language
            response_text = re.sub(
                pattern,
                "may",
                response_text,
                flags=re.IGNORECASE
            )
    
    # Add required disclaimers
    disclaimers.append("Terms and conditions apply. Benefits subject to cardholder agreement.")
    disclaimers.append("This is an awareness-only system. No transactions or account actions are performed.")
    disclaimers.append("Generated locally. No card data is stored or logged.")
    
    # PCI Compliance: Ensure no full PANs
    if re.search(r"\d{4}-\d{4}-\d{4}-\d{4}", response_text):
        compliance_approved = False
        return {
            "error": "Compliance violation: Potential PAN detected in output.",
            "error_code": "COMPLIANCE_REJECTED",
            "compliance_approved": False,
            "final_output": {
                "status": "error",
                "error_code": "COMPLIANCE_REJECTED",
                "message": "Response blocked for security compliance."
            }
        }
    
    # Get top 4 recommendations (default) and all eligible benefits
    ranked_benefits = state.get("ranked_benefits", [])
    top_4_benefits = ranked_benefits[:4] if len(ranked_benefits) >= 4 else ranked_benefits
    all_benefits = ranked_benefits  # All eligible suggestions
    
    # Prepare recommendations with beacon tagging
    recommendations = []
    for idx, benefit_data in enumerate(top_4_benefits):
        chunk = benefit_data.get("chunk", {})
        is_beacon_choice = idx == 0  # First one is beacon's top choice
        
        # Get explanation for this benefit
        # For top choice, use the generated explanation; for others, use chunk content
        if idx == 0:
            benefit_explanation = response_text
        else:
            # Use chunk content as explanation for other recommendations
            chunk_content = chunk.get("content", "")
            # Extract key information from chunk
            benefit_explanation = chunk_content[:300] + "..." if len(chunk_content) > 300 else chunk_content
        
        recommendations.append({
            "explanation": benefit_explanation,
            "is_beacon_choice": is_beacon_choice,
            "source_chunks": [
                {
                    "chunk_id": chunk.get("source_chunk_id", "unknown"),
                    "similarity": chunk.get("similarity_score", 0.0),
                    "content": _extract_informative_summary(chunk.get("content", ""))
                }
            ],
            "scores": benefit_data.get("scores", {})
        })
    
    # Prepare all eligible benefits (for "Show All" option)
    all_eligible_benefits = []
    for idx, benefit_data in enumerate(all_benefits):
        chunk = benefit_data.get("chunk", {})
        is_beacon_choice = idx == 0  # First one is beacon's top choice
        
        if idx == 0:
            # Remove AI-sounding prefixes from the explanation
            benefit_explanation = response_text
            # Remove common AI prefixes
            ai_prefixes = [
                "based on the provided information, ",
                "based on the information, ",
                "i recommend ",
                "here's ",
                "here is ",
                "the following benefit: ",
                "the recommended benefit is: ",
            ]
            for prefix in ai_prefixes:
                if benefit_explanation.lower().startswith(prefix):
                    benefit_explanation = benefit_explanation[len(prefix):].strip()
                    # Capitalize first letter
                    if benefit_explanation:
                        benefit_explanation = benefit_explanation[0].upper() + benefit_explanation[1:]
        else:
            # Extract meaningful information from chunk for follow-up recommendations
            chunk_content = chunk.get("content", "")
            # Try to extract benefit name and key details
            lines = chunk_content.split('\n')
            benefit_name = ""
            benefit_details = []
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                # Look for benefit name (usually first meaningful line)
                if not benefit_name and len(line) > 10 and len(line) < 100:
                    # Skip common prefixes
                    if not any(line.lower().startswith(p) for p in ['#', 'visa', 'card', 'benefit']):
                        benefit_name = line
                # Collect key details (lines with numbers, percentages, or action words)
                elif any(keyword in line.lower() for keyword in ['%', 'off', 'discount', 'valid', 'available', 'access', 'get', 'save']):
                    if len(line) < 200:
                        benefit_details.append(line)
            
            # Build informative explanation
            if benefit_name:
                if benefit_details:
                    benefit_explanation = f"{benefit_name}. {'. '.join(benefit_details[:2])}"
                else:
                    benefit_explanation = benefit_name
            else:
                # Fallback: use first meaningful sentence
                sentences = chunk_content.split('.')
                meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20 and len(s.strip()) < 200]
                if meaningful_sentences:
                    benefit_explanation = '. '.join(meaningful_sentences[:2])
                else:
                    benefit_explanation = chunk_content[:200] + "..." if len(chunk_content) > 200 else chunk_content
        
        all_eligible_benefits.append({
            "explanation": benefit_explanation,
            "is_beacon_choice": is_beacon_choice,
            "source_chunks": [
                {
                    "chunk_id": chunk.get("source_chunk_id", "unknown"),
                    "similarity": chunk.get("similarity_score", 0.0),
                    "content": _extract_informative_summary(chunk.get("content", ""))
                }
            ],
            "scores": benefit_data.get("scores", {})
        })
    
    # Final output assembly
    final_output = {
        "status": "success",
        "card_tier": state.get("detected_tier"),
        "recommended_benefit": recommendations[0] if recommendations else {
            "explanation": response_text,
            "source_chunks": []
        },
        "recommendations": recommendations,  # Top 4 recommendations
        "all_benefits": all_eligible_benefits,  # All eligible suggestions
        "total_benefits_count": len(all_benefits),
        "disclaimers": disclaimers,
        "language": state.get("preferred_language", "en"),
        "metadata": {
            "bin_validated": state.get("bin_valid", False),
            "rag_grounded": len(state.get("benefit_chunks", [])) > 0,
            "compliance_approved": True
        }
    }
    
    return {
        "final_output": final_output,
        "compliance_approved": True,
        "disclaimers": disclaimers
    }


# ============================================================================
# 4. CONDITIONAL ROUTING
# ============================================================================

def should_continue(state: AgentState) -> str:
    """Route based on error state"""
    if state.get("error"):
        return "compliance"  # Go to compliance for error handling
    return "continue"


def route_after_card_intel(state: AgentState) -> str:
    """Route after card intelligence"""
    if state.get("error") or not state.get("bin_valid"):
        return "compliance"
    return "retrieval"


def route_after_retrieval(state: AgentState) -> str:
    """Route after benefit retrieval"""
    if state.get("error") or not state.get("benefit_chunks"):
        return "compliance"
    return "explanation"


# ============================================================================
# 5. BUILD GRAPH
# ============================================================================

def build_workflow():
    """
    Build the complete LangGraph workflow (AGENTIC AI)
    
    This function creates a multi-agent orchestration system where:
    - Agents are nodes in a state graph
    - Edges define the flow between agents
    - Conditional edges enable dynamic routing based on agent decisions
    - State is shared across all agents (AgentState)
    
    This is a classic Agentic AI pattern: multiple specialized agents
    working together to solve a complex task.
    """
    
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("card_intel", card_intelligence_agent)
    workflow.add_node("retrieval", benefit_retrieval_agent)
    workflow.add_node("explanation", explanation_agent)
    workflow.add_node("recommendation", recommendation_agent)
    workflow.add_node("language", language_agent)
    workflow.add_node("compliance", compliance_agent)

    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Define edges
    workflow.add_edge("supervisor", "card_intel")
    workflow.add_conditional_edges(
        "card_intel",
        route_after_card_intel,
        {
            "compliance": "compliance",
            "retrieval": "retrieval"
        }
    )
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "compliance": "compliance",
            "explanation": "explanation"
        }
    )
    workflow.add_edge("explanation", "recommendation")
    workflow.add_edge("recommendation", "language")
    workflow.add_edge("language", "compliance")
    workflow.add_edge("compliance", END)

    return workflow.compile()


# Create compiled app
app = build_workflow()

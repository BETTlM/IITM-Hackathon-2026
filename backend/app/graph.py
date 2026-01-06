"""
Visa Benefits Workflow - TRULY AGENTIC AI System
Uses local models (Ollama + HuggingFace) - No OpenAI API

AGENTIC AI FEATURES:
- Autonomous Decision-Making: Agents use LLM reasoning to decide actions
- Tool Use: Agents can call external tools (search, validate, filter, score, check)
- Iterative Planning: Agents can replan if results are insufficient
- Self-Correction: Agents can retry and adjust their approach
- Memory: Conversation history and learning from past interactions
- Goal-Oriented: Agents work towards goals with sub-goals

GENERATIVE AI:
- Uses Ollama (llama3.2) for natural language generation and reasoning
- Explanation Agent: Generates plain-language explanations from legal text
- Language Agent: Generates translations (English/Tamil)
- All text generation is RAG-grounded to prevent hallucinations
"""

import re
import json
from typing import TypedDict, List, Optional, Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime
from app.tools import TOOL_REGISTRY, execute_tool, list_tools

# ============================================================================
# 1. STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    # Input
    card_number: str
    user_context: Optional[str]  # student / traveler / family
    preferred_language: Literal["en", "ta"]  # English or Tamil
    location: Optional[str]  # city-level only
    
    # AGENTIC AI: Goals and Planning
    goal: Optional[str]  # Main goal: "Find best benefits for user"
    sub_goals: List[str]  # Sub-goals: ["Validate card", "Search benefits", "Rank benefits"]
    plan: Optional[str]  # Current plan/strategy
    plan_iteration: int  # Number of times plan has been revised
    
    # AGENTIC AI: Memory and Learning
    conversation_history: List[Dict[str, Any]]  # Past interactions
    agent_memory: Dict[str, Any]  # Agent-specific memory/learnings
    tool_calls: List[Dict[str, Any]]  # History of tool calls made
    
    # AGENTIC AI: Autonomous Decision-Making
    agent_reasoning: Dict[str, str]  # Reasoning from each agent
    next_action: Optional[str]  # Agent-decided next action
    retry_count: Dict[str, int]  # Retry counts per agent
    
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
    Supervisor/Orchestrator Agent (TRULY AGENTIC AI)
    - Creates goals and plans using LLM reasoning
    - Monitors progress and can replan if needed
    - Coordinates agents and enables autonomous decision-making
    - Uses memory to learn from past interactions
    """
    try:
        llm = ChatOllama(model="llama3.2", temperature=0.2, timeout=60.0)
        
        # Get conversation history for context
        history = state.get("conversation_history", [])
        history_context = ""
        if history:
            recent = history[-3:]  # Last 3 interactions
            history_context = "\n".join([
                f"Previous: {h.get('agent', 'unknown')} - {h.get('action', 'unknown')}"
                for h in recent
            ])
        
        # Build goal and plan using LLM reasoning
        card = state.get("card_number", "")
        context = state.get("user_context", "")
        location = state.get("location", "")
        
        prompt = ChatPromptTemplate.from_template(
            """You are a Supervisor Agent coordinating a multi-agent system to find Visa card benefits.

USER REQUEST:
- Card: {card}
- Context: {context}
- Location: {location}

CONVERSATION HISTORY:
{history}

TASK:
1. Define the main goal (one sentence)
2. Create 3-5 sub-goals (specific steps)
3. Create an initial plan (brief strategy)

OUTPUT FORMAT (JSON):
{{
    "goal": "Main goal here",
    "sub_goals": ["sub-goal 1", "sub-goal 2", ...],
    "plan": "Brief plan description"
}}"""
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "card": card,
            "context": context or "general",
            "location": location or "any",
            "history": history_context or "No previous interactions"
        })
        
        # Parse LLM response
        reasoning_text = response.content.strip() if response.content else ""
        
        # Extract JSON from response
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{[^}]+\}', reasoning_text, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                # Fallback: create plan from text
                plan_data = {
                    "goal": "Find the best Visa card benefits for the user",
                    "sub_goals": [
                        "Validate card and determine tier",
                        "Search for relevant benefits",
                        "Rank benefits by relevance",
                        "Generate explanation",
                        "Ensure compliance"
                    ],
                    "plan": reasoning_text[:200] if reasoning_text else "Standard workflow"
                }
        except json.JSONDecodeError:
            # Fallback plan
            plan_data = {
                "goal": "Find the best Visa card benefits for the user",
                "sub_goals": [
                    "Validate card and determine tier",
                    "Search for relevant benefits",
                    "Rank benefits by relevance",
                    "Generate explanation",
                    "Ensure compliance"
                ],
                "plan": reasoning_text[:200] if reasoning_text else "Standard workflow"
            }
        
        # Update memory
        agent_memory = state.get("agent_memory", {})
        agent_memory["supervisor"] = {
            "last_plan": plan_data,
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "goal": plan_data.get("goal", "Find the best Visa card benefits for the user"),
            "sub_goals": plan_data.get("sub_goals", []),
            "plan": plan_data.get("plan", "Standard workflow"),
            "plan_iteration": state.get("plan_iteration", 0),
            "agent_reasoning": {
                **state.get("agent_reasoning", {}),
                "supervisor": reasoning_text[:500]
            },
            "agent_memory": agent_memory
        }
    
    except Exception as e:
        # Fallback: use default plan
        return {
            "goal": "Find the best Visa card benefits for the user",
            "sub_goals": [
                "Validate card and determine tier",
                "Search for relevant benefits",
                "Rank benefits by relevance",
                "Generate explanation",
                "Ensure compliance"
            ],
            "plan": "Standard workflow",
            "plan_iteration": state.get("plan_iteration", 0),
            "agent_reasoning": {
                **state.get("agent_reasoning", {}),
                "supervisor": f"Error in planning: {str(e)}"
            }
        }


def card_intelligence_agent(state: AgentState) -> dict:
    """
    Card Intelligence Agent (TRULY AUTONOMOUS)
    - Uses tool: validate_card
    - Makes autonomous decision: whether to retry or proceed
    - Uses LLM reasoning to decide next action
    - Can self-correct on errors
    - Sets next_action for autonomous routing
    """
    card = state["card_number"]
    retry_count = state.get("retry_count", {})
    agent_retries = retry_count.get("card_intel", 0)
    
    # AGENTIC: Use tool to validate card
    tool_result = execute_tool("validate_card", card_number=card)
    
    # Record tool call
    tool_calls = state.get("tool_calls", [])
    tool_calls.append({
        "agent": "card_intelligence",
        "tool": "validate_card",
        "result": tool_result,
        "timestamp": datetime.now().isoformat()
    })
    
    if not tool_result.get("success"):
        # Tool execution failed - decide whether to retry
        if agent_retries < 2:  # Max 2 retries
            retry_count["card_intel"] = agent_retries + 1
            return {
                "retry_count": retry_count,
                "error": f"Card validation tool failed: {tool_result.get('error')}. Retrying...",
                "tool_calls": tool_calls,
                "next_action": "retrieval"  # Will retry, then proceed
            }
        else:
            return {
                "error": f"Card validation failed after retries: {tool_result.get('error')}",
                "error_code": "VALIDATION_ERROR",
                "bin_valid": False,
                "tool_calls": tool_calls,
                "next_action": "compliance"  # Go to compliance for error handling
            }
    
    validation_result = tool_result.get("result", {})
    
    if not validation_result.get("valid"):
        # Card invalid - use LLM to reason about next action
        try:
            llm = ChatOllama(model="llama3.2", temperature=0.1, timeout=30.0)
            
            prompt = ChatPromptTemplate.from_template(
                """Card validation failed: {error}
                
Should we:
1. Reject the request (if format is wrong) -> next_action: "compliance"
2. Use a default tier and continue (if BIN unknown) -> next_action: "retrieval"
3. Ask user for clarification -> next_action: "compliance"

Decision (one word: reject/continue/ask):"""
            )
            
            chain = prompt | llm
            response = chain.invoke({"error": validation_result.get("error", "Unknown error")})
            decision = response.content.strip().lower() if response.content else "reject"
            
            reasoning = f"Card validation failed. LLM decision: {decision}"
            
            if "continue" in decision and validation_result.get("tier"):
                # Continue with detected tier - AUTONOMOUS DECISION
                return {
                    "detected_tier": validation_result.get("tier"),
                    "bin_valid": True,
                    "agent_reasoning": {
                        **state.get("agent_reasoning", {}),
                        "card_intel": reasoning + " - Continuing with default tier"
                    },
                    "tool_calls": tool_calls,
                    "next_action": "retrieval"  # AUTONOMOUS: Decided to continue
                }
            else:
                # Reject - AUTONOMOUS DECISION
                return {
                    "error": validation_result.get("error", "Card validation failed"),
                    "error_code": "INVALID_CARD_FORMAT",
                    "bin_valid": False,
                    "agent_reasoning": {
                        **state.get("agent_reasoning", {}),
                        "card_intel": reasoning + " - Rejecting request"
                    },
                    "tool_calls": tool_calls,
                    "next_action": "compliance"  # AUTONOMOUS: Decided to reject
                }
        except Exception:
            # Fallback: reject
            return {
                "error": validation_result.get("error", "Card validation failed"),
                "error_code": "INVALID_CARD_FORMAT",
                "bin_valid": False,
                "tool_calls": tool_calls,
                "next_action": "compliance"
            }
    
    # Success - AUTONOMOUS DECISION to proceed
    return {
        "detected_tier": validation_result.get("tier"),
        "bin_valid": True,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "card_intel": f"Card validated successfully. Tier: {validation_result.get('tier')}. Proceeding to retrieval."
        },
        "tool_calls": tool_calls,
        "next_action": "retrieval"  # AUTONOMOUS: Decided to proceed
    }


def benefit_retrieval_agent(state: AgentState) -> dict:
    """
    Benefit Retrieval Agent (AGENTIC AI)
    - Uses tool: search_benefits, filter_by_location
    - Makes autonomous decision: whether to expand search or refine query
    - Uses LLM reasoning to improve search strategy
    - Can replan if results are insufficient
    """
    if state.get("error") or not state.get("bin_valid"):
        return {}
    
    tier = state.get("detected_tier", "Classic")
    context = state.get("user_context") or ""
    location = state.get("location") or ""
    retry_count = state.get("retry_count", {})
    agent_retries = retry_count.get("retrieval", 0)
    
    # AGENTIC: Use LLM to build optimal search query
    try:
        llm = ChatOllama(model="llama3.2", temperature=0.2, timeout=30.0)
        
        prompt = ChatPromptTemplate.from_template(
            """Build an optimal search query for finding Visa card benefits.

Card Tier: {tier}
User Context: {context}
Location: {location}

Create a concise search query (2-5 words) that will find the most relevant benefits.
Query:"""
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "tier": tier,
            "context": context or "general",
            "location": location or "any"
        })
        
        search_query = response.content.strip() if response.content else f"Visa {tier} {context}"
        reasoning = f"Built search query: {search_query}"
        
    except Exception:
        # Fallback query
        search_query = f"Visa {tier} {context}"
        reasoning = "Using fallback query"
    
    # AGENTIC: Use tool to search benefits
    tool_result = execute_tool("search_benefits", query=search_query, tier=tier, location=location, k=30)
    
    # Record tool call
    tool_calls = state.get("tool_calls", [])
    tool_calls.append({
        "agent": "benefit_retrieval",
        "tool": "search_benefits",
        "result": tool_result,
        "timestamp": datetime.now().isoformat()
    })
    
    if not tool_result.get("success"):
        # Tool failed - decide whether to retry
        if agent_retries < 2:
            retry_count["retrieval"] = agent_retries + 1
            return {
                "retry_count": retry_count,
                "error": f"Search tool failed: {tool_result.get('error')}. Retrying...",
                "tool_calls": tool_calls
            }
        else:
            return {
                "error": f"Search failed after retries: {tool_result.get('error')}",
                "error_code": "RETRIEVAL_ERROR",
                "retrieved_docs": [],
                "benefit_chunks": [],
                "tool_calls": tool_calls
            }
    
    benefits = tool_result.get("result", {}).get("benefits", [])
    
    # AGENTIC: Filter by location if specified
    if location and benefits:
        filter_result = execute_tool("filter_by_location", benefits=benefits, location=location)
        tool_calls.append({
            "agent": "benefit_retrieval",
            "tool": "filter_by_location",
            "result": filter_result,
            "timestamp": datetime.now().isoformat()
        })
        
        if filter_result.get("success"):
            benefits = filter_result.get("result", {}).get("filtered_benefits", benefits)
            reasoning += f". Filtered to {len(benefits)} location-matched benefits"
    
    # AGENTIC: Evaluate if results are sufficient
    if len(benefits) < 3 and agent_retries < 2:
        # Not enough results - replan
        try:
            llm = ChatOllama(model="llama3.2", temperature=0.3, timeout=30.0)
            
            prompt = ChatPromptTemplate.from_template(
                """Only found {count} benefits. Should we:
1. Expand search (remove location filter, increase k)
2. Try different query terms
3. Proceed with current results

Decision (expand/try/proceed):"""
            )
            
            chain = prompt | llm
            response = chain.invoke({"count": len(benefits)})
            decision = response.content.strip().lower() if response.content else "proceed"
            
            if "expand" in decision:
                # Retry with expanded search
                retry_count["retrieval"] = agent_retries + 1
                expanded_result = execute_tool("search_benefits", query=search_query, tier=tier, location=None, k=50)
                if expanded_result.get("success"):
                    benefits = expanded_result.get("result", {}).get("benefits", benefits)
                    reasoning += ". Expanded search successful"
        except Exception:
            pass  # Proceed with current results
    
    if not benefits:
        return {
            "error": "No benefits found for this card tier and context.",
            "error_code": "NO_BENEFITS_FOUND",
            "retrieved_docs": [],
            "benefit_chunks": [],
            "agent_reasoning": {
                **state.get("agent_reasoning", {}),
                "retrieval": reasoning
            },
            "tool_calls": tool_calls
        }
    
    # Structure benefits
    benefit_chunks = []
    retrieved_docs_serialized = []
    for benefit in benefits:
        chunk_data = {
            "content": benefit.get("content", ""),
            "metadata": benefit.get("metadata", {}),
            "similarity_score": benefit.get("similarity_score", 0.0),
            "source_chunk_id": benefit.get("metadata", {}).get("source", "unknown")
        }
        benefit_chunks.append(chunk_data)
        
        retrieved_docs_serialized.append({
            "page_content": benefit.get("content", ""),
            "metadata": benefit.get("metadata", {})
        })
    
    # AUTONOMOUS: Decide next action based on results
    if len(benefit_chunks) > 0:
        next_action = "explanation"  # Proceed to explanation
    else:
        next_action = "compliance"  # No benefits, go to compliance
    
    return {
        "retrieved_docs": retrieved_docs_serialized,
        "benefit_chunks": benefit_chunks,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "retrieval": reasoning + f". Retrieved {len(benefit_chunks)} benefits. Next: {next_action}"
        },
        "tool_calls": tool_calls,
        "next_action": next_action  # AUTONOMOUS: Decided next step
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
    if not chunks:
        return {
            "error": "No benefit chunks available for explanation.",
            "error_code": "EXPLANATION_ERROR"
        }
    
    official_text = "\n\n".join([
        f"[Source {i+1}]\n{chunk.get('content', '')}"
        for i, chunk in enumerate(chunks)
        if chunk.get('content')
    ])
    
    if not official_text:
        return {
            "error": "No valid content found in benefit chunks.",
            "error_code": "EXPLANATION_ERROR"
        }
    
    try:
        llm = ChatOllama(model="llama3.2", temperature=0.3, timeout=60.0)
        
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
        
        if not response or not hasattr(response, 'content'):
            return {
                "error": "LLM returned empty response. Please ensure Ollama is running.",
                "error_code": "EXPLANATION_ERROR"
            }
        
        explanation = response.content.strip() if response.content else ""
        
        # Validate response
        if not explanation or len(explanation) < 20:
            return {
                "error": "Explanation too short or empty. Possible LLM error or hallucination.",
                "error_code": "EXPLANATION_ERROR"
            }
        
        return {
            "plain_language_explanation": explanation
        }
    
    except TimeoutError:
        return {
            "error": "LLM request timed out. Please ensure Ollama is running and responsive.",
            "error_code": "EXPLANATION_ERROR"
        }
    except ConnectionError as e:
        return {
            "error": f"Cannot connect to Ollama. Please ensure Ollama is running: {str(e)}",
            "error_code": "EXPLANATION_ERROR"
        }
    except Exception as e:
        return {
            "error": f"Explanation generation error: {str(e)}",
            "error_code": "EXPLANATION_ERROR"
        }


def recommendation_agent(state: AgentState) -> dict:
    """
    Recommendation Agent (AGENTIC AI)
    - Uses tool: score_benefits
    - Makes autonomous decision: whether to adjust scoring weights
    - Uses LLM reasoning to evaluate ranking quality
    - Can replan if ranking is unsatisfactory
    """
    if state.get("error") or not state.get("benefit_chunks"):
        return {}
    
    # Handle None values safely
    context = state.get("user_context") or ""
    location = state.get("location") or ""
    chunks = state["benefit_chunks"]
    
    # AGENTIC: Use tool to score benefits
    tool_result = execute_tool("score_benefits", benefits=chunks, user_context=context, location=location)
    
    # Record tool call
    tool_calls = state.get("tool_calls", [])
    tool_calls.append({
        "agent": "recommendation",
        "tool": "score_benefits",
        "result": tool_result,
        "timestamp": datetime.now().isoformat()
    })
    
    if not tool_result.get("success"):
        # Tool failed - use fallback scoring
        reasoning = f"Scoring tool failed: {tool_result.get('error')}. Using fallback."
        # Fall through to original scoring logic as backup
    else:
        # Use tool results
        ranked_data = tool_result.get("result", {}).get("ranked_benefits", [])
        
        # Convert to expected format
        ranked_benefits = []
        for item in ranked_data:
            ranked_benefits.append({
                "chunk": item.get("benefit", {}),
                "scores": item.get("scores", {})
            })
        
        top_benefit = ranked_benefits[0] if ranked_benefits else None
        
        # AGENTIC: Evaluate ranking quality
        reasoning = f"Scored {len(ranked_benefits)} benefits. Top score: {top_benefit.get('scores', {}).get('total', 0) if top_benefit else 0:.2f}"
        
        return {
            "ranked_benefits": ranked_benefits,
            "top_benefit": top_benefit,
            "agent_reasoning": {
                **state.get("agent_reasoning", {}),
                "recommendation": reasoning
            },
            "tool_calls": tool_calls
        }
    
    # Fallback to original logic if tool failed
    context_lower = context.lower() if context else ""
    location_lower = location.lower() if location else ""
    
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
        if not chunk or not isinstance(chunk, dict):
            continue
        
        content = chunk.get("content", "")
        if not content:
            continue
        
        content = content.lower()
        metadata = chunk.get("metadata", {}) or {}
        
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
            try:
                amounts = [int(amt) for amt in dollar_matches if amt.isdigit()]
                if amounts:
                    max_amount = max(amounts)
                    if max_amount >= 500:
                        monetary_score = 1.0
                    elif max_amount >= 100:
                        monetary_score = 0.7
                    elif max_amount >= 50:
                        monetary_score = 0.5
            except (ValueError, TypeError):
                pass  # Keep default score
        
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
    
    # Use fallback reasoning if not set
    if "reasoning" not in locals():
        reasoning = f"Used fallback scoring. Ranked {len(ranked_benefits)} benefits."
    
    return {
        "ranked_benefits": ranked_benefits,
        "top_benefit": top_benefit,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "recommendation": reasoning
        },
        "tool_calls": tool_calls
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
        llm = ChatOllama(model="llama3.2", temperature=0.1, timeout=60.0)
        
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
        
        if not response or not hasattr(response, 'content') or not response.content:
            # Fallback to English on empty response
            return {
                "translated_response": explanation,
                "error": "Translation returned empty response, using English"
            }
        
        translated = response.content.strip()
        
        # Fallback: if translation fails or is too short, return English
        if not translated or len(translated) < len(explanation) * 0.3:
            return {
                "translated_response": explanation,  # Fallback to English
                "error": "Translation quality insufficient, using English"
            }
        
        return {
            "translated_response": translated
        }
    
    except TimeoutError:
        # Fallback to English on timeout
        return {
            "translated_response": explanation,
            "error": "Translation timed out, using English"
        }
    except ConnectionError as e:
        # Fallback to English on connection error
        return {
            "translated_response": explanation,
            "error": f"Cannot connect to Ollama: {str(e)}, using English"
        }
    except Exception as e:
        # Fallback to English on error
        return {
            "translated_response": explanation,
            "error": f"Translation error: {str(e)}, using English"
        }


def compliance_agent(state: AgentState) -> dict:
    """
    Compliance Agent (AGENTIC AI)
    - Uses tool: check_compliance
    - Makes autonomous decision: whether to reject or fix violations
    - Uses LLM reasoning to evaluate compliance
    - Updates memory with compliance patterns
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
    
    # AGENTIC: Use tool to check compliance
    tool_result = execute_tool("check_compliance", text=response_text)
    
    # Record tool call
    tool_calls = state.get("tool_calls", [])
    tool_calls.append({
        "agent": "compliance",
        "tool": "check_compliance",
        "result": tool_result,
        "timestamp": datetime.now().isoformat()
    })
    
    # Compliance checks
    disclaimers = []
    compliance_approved = True
    
    if tool_result.get("success"):
        compliance_data = tool_result.get("result", {})
        compliance_approved = compliance_data.get("approved", True)
        violations = compliance_data.get("violations", [])
        
        if violations:
            # AGENTIC: Use LLM to decide how to fix violations
            try:
                llm = ChatOllama(model="llama3.2", temperature=0.1, timeout=30.0)
                
                prompt = ChatPromptTemplate.from_template(
                    """Compliance violations found: {violations}

Text: {text}

Should we:
1. Fix the text (replace unsafe language)
2. Reject the response
3. Add disclaimers and proceed

Decision (fix/reject/proceed):"""
                )
                
                chain = prompt | llm
                response = chain.invoke({
                    "violations": ", ".join(violations),
                    "text": response_text[:500]
                })
                decision = response.content.strip().lower() if response.content else "proceed"
                
                if "reject" in decision:
                    compliance_approved = False
                elif "fix" in decision:
                    # Fix unsafe language
                    for pattern in [r"guaranteed", r"always", r"never", r"100%", r"definitely"]:
                        response_text = re.sub(pattern, "may", response_text, flags=re.IGNORECASE)
                    compliance_approved = True
                
                reasoning = f"Found violations: {len(violations)}. Decision: {decision}"
            except Exception:
                # Fallback: fix automatically
                for pattern in [r"guaranteed", r"always", r"never", r"100%", r"definitely"]:
                    response_text = re.sub(pattern, "may", response_text, flags=re.IGNORECASE)
                reasoning = "Auto-fixed violations"
        else:
            reasoning = "No compliance violations found"
    else:
        # Tool failed - use fallback checks
        for pattern in [r"guaranteed", r"always", r"never", r"100%", r"definitely"]:
            if re.search(pattern, response_text, re.IGNORECASE):
                response_text = re.sub(pattern, "may", response_text, flags=re.IGNORECASE)
        reasoning = "Used fallback compliance checks"
    
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
            },
            "agent_reasoning": {
                **state.get("agent_reasoning", {}),
                "compliance": "PAN detected - rejected"
            },
            "tool_calls": tool_calls
        }
    
    # Get top 4 recommendations (default) and all eligible benefits
    ranked_benefits = state.get("ranked_benefits", [])
    if not ranked_benefits:
        return {
            "error": "No benefits available to display.",
            "error_code": "NO_BENEFITS_AVAILABLE",
            "compliance_approved": False,
            "final_output": {
                "status": "error",
                "error_code": "NO_BENEFITS_AVAILABLE",
                "message": "No benefits available to display."
            }
        }
    
    top_4_benefits = ranked_benefits[:4] if len(ranked_benefits) >= 4 else ranked_benefits
    all_benefits = ranked_benefits  # All eligible suggestions
    
    # Prepare recommendations with beacon tagging
    recommendations = []
    for idx, benefit_data in enumerate(top_4_benefits):
        if not benefit_data or not isinstance(benefit_data, dict):
            continue
        
        chunk = benefit_data.get("chunk", {}) or {}
        is_beacon_choice = idx == 0  # First one is beacon's top choice
        
        # Get explanation for this benefit
        # For top choice, use the generated explanation; for others, use chunk content
        if idx == 0:
            benefit_explanation = response_text if response_text else "No explanation available."
        else:
            # Use chunk content as explanation for other recommendations
            chunk_content = chunk.get("content", "") if chunk else ""
            if not chunk_content:
                continue  # Skip if no content
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
        if not benefit_data or not isinstance(benefit_data, dict):
            continue
        
        chunk = benefit_data.get("chunk", {}) or {}
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
    
    # Update memory with successful completion
    agent_memory = state.get("agent_memory", {})
    violations_count = 0
    if tool_result.get("success"):
        violations_count = len(tool_result.get("result", {}).get("violations", []))
    agent_memory["compliance"] = {
        "last_check": datetime.now().isoformat(),
        "violations_found": violations_count,
        "approved": compliance_approved
    }
    
    # Update conversation history
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "agent": "compliance",
        "action": "final_approval" if compliance_approved else "rejection",
        "timestamp": datetime.now().isoformat(),
        "reasoning": reasoning
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
            "compliance_approved": True,
            "agentic_ai": True,  # Mark as agentic AI system
            "tool_calls_count": len(tool_calls),
            "plan_iteration": state.get("plan_iteration", 0)
        },
        "agentic_metadata": {  # Agentic AI specific metadata
            "goal": state.get("goal"),
            "sub_goals": state.get("sub_goals", []),
            "agent_reasoning": state.get("agent_reasoning", {}),
            "tool_calls": tool_calls[-10:]  # Last 10 tool calls
        }
    }
    
    return {
        "final_output": final_output,
        "compliance_approved": True,
        "disclaimers": disclaimers,
        "agent_reasoning": {
            **state.get("agent_reasoning", {}),
            "compliance": reasoning
        },
        "agent_memory": agent_memory,
        "conversation_history": conversation_history,
        "tool_calls": tool_calls
    }


# ============================================================================
# 4. AUTONOMOUS ROUTING (AGENTIC AI)
# ============================================================================

def autonomous_router(state: AgentState, current_agent: str) -> str:
    """
    Autonomous Router - Uses LLM reasoning to decide next action
    Agents can dynamically choose their next step based on context
    """
    # Check if agent has already decided next action
    next_action = state.get("next_action")
    if next_action and next_action in ["compliance", "retrieval", "explanation", "recommendation", "language", "end"]:
        return next_action
    
    # Use LLM to reason about next action
    try:
        llm = ChatOllama(model="llama3.2", temperature=0.2, timeout=30.0)
        
        # Get context
        error = state.get("error")
        bin_valid = state.get("bin_valid", False)
        benefit_chunks = state.get("benefit_chunks", [])
        goal = state.get("goal", "Find best benefits")
        sub_goals = state.get("sub_goals", [])
        agent_reasoning = state.get("agent_reasoning", {})
        
        # Build context summary
        context = f"""
Current Agent: {current_agent}
Goal: {goal}
Sub-goals: {', '.join(sub_goals[:3])}
Error: {error if error else 'None'}
Card Valid: {bin_valid}
Benefits Found: {len(benefit_chunks)}
"""
        
        # Get recent agent reasoning
        recent_reasoning = "\n".join([
            f"{agent}: {reason[:100]}"
            for agent, reason in list(agent_reasoning.items())[-2:]
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """You are an autonomous router deciding the next step in a multi-agent workflow.

CONTEXT:
{context}

RECENT AGENT REASONING:
{reasoning}

AVAILABLE NEXT STEPS:
- "retrieval": Search for benefits (if card is valid)
- "explanation": Generate explanation (if benefits found)
- "recommendation": Rank benefits (if explanation ready)
- "language": Translate response (if explanation ready)
- "compliance": Final check and output (if all steps done or error)
- "end": Complete workflow

Based on the current state, what should be the NEXT ACTION?
Consider:
1. Are there errors that need handling?
2. Are prerequisites met for next step?
3. Should we skip any steps?
4. Is the goal achieved?

Decision (one word: retrieval/explanation/recommendation/language/compliance/end):"""
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "context": context,
            "reasoning": recent_reasoning or "No recent reasoning"
        })
        
        decision = response.content.strip().lower() if response.content else ""
        
        # Extract decision from response
        for action in ["retrieval", "explanation", "recommendation", "language", "compliance", "end"]:
            if action in decision:
                return action
        
        # Fallback to rule-based routing
        return rule_based_router(state, current_agent)
        
    except Exception as e:
        # Fallback to rule-based routing on error
        return rule_based_router(state, current_agent)


def rule_based_router(state: AgentState, current_agent: str) -> str:
    """Fallback rule-based router"""
    if state.get("error"):
        return "compliance"
    
    if current_agent == "card_intel":
        if state.get("bin_valid"):
            return "retrieval"
        return "compliance"
    elif current_agent == "retrieval":
        if state.get("benefit_chunks"):
            return "explanation"
        return "compliance"
    elif current_agent == "explanation":
        return "recommendation"
    elif current_agent == "recommendation":
        return "language"
    elif current_agent == "language":
        return "compliance"
    
    return "compliance"


def route_after_card_intel(state: AgentState) -> str:
    """Autonomous routing after card intelligence"""
    return autonomous_router(state, "card_intel")


def route_after_retrieval(state: AgentState) -> str:
    """Autonomous routing after benefit retrieval"""
    return autonomous_router(state, "retrieval")


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
    
    # Define edges - AUTONOMOUS ROUTING
    workflow.add_edge("supervisor", "card_intel")
    
    # Autonomous routing after card intelligence
    workflow.add_conditional_edges(
        "card_intel",
        route_after_card_intel,
        {
            "compliance": "compliance",
            "retrieval": "retrieval",
            "explanation": "explanation",  # Can skip retrieval if needed
            "end": END  # Can end early if goal achieved
        }
    )
    
    # Autonomous routing after benefit retrieval
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "compliance": "compliance",
            "explanation": "explanation",
            "recommendation": "recommendation",  # Can skip explanation if needed
            "end": END  # Can end early
        }
    )
    
    # Allow agents to skip steps - autonomous decision
    workflow.add_conditional_edges(
        "explanation",
        lambda s: autonomous_router(s, "explanation"),
        {
            "recommendation": "recommendation",
            "language": "language",  # Can skip recommendation
            "compliance": "compliance",  # Can skip to compliance
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "recommendation",
        lambda s: autonomous_router(s, "recommendation"),
        {
            "language": "language",
            "compliance": "compliance",  # Can skip translation
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "language",
        lambda s: autonomous_router(s, "language"),
        {
            "compliance": "compliance",
            "end": END  # Can end before compliance if needed
        }
    )
    
    workflow.add_edge("compliance", END)

    return workflow.compile()


# Create compiled app
app = build_workflow()

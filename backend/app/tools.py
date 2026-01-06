"""
Tool Registry for Agentic AI System
Agents can autonomously call these tools to accomplish their goals
"""

import re
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


class Tool:
    """Base class for tools that agents can use"""
    
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool"""
        try:
            result = self.func(**kwargs)
            return {
                "success": True,
                "result": result,
                "tool": self.name
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool": self.name
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LLM understanding"""
        return {
            "name": self.name,
            "description": self.description
        }


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def search_benefits_tool(query: str, tier: str, location: Optional[str] = None, k: int = 20) -> Dict[str, Any]:
    """
    Search for benefits in the vector database
    Returns: List of benefit chunks with similarity scores
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        db = Chroma(
            persist_directory="./visa_db",
            embedding_function=embeddings
        )
        
        # Build enhanced query
        query_parts = [f"Visa {tier}", query]
        if location:
            query_parts.append(location)
        
        search_query = " ".join(query_parts)
        
        # Perform search
        results = db.similarity_search_with_score(search_query, k=k)
        
        # Structure results
        benefits = []
        for doc, score in results:
            benefits.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score)
            })
        
        return {
            "benefits": benefits,
            "count": len(benefits),
            "query": search_query
        }
    except Exception as e:
        raise Exception(f"Search failed: {str(e)}")


def validate_card_tool(card_number: str) -> Dict[str, Any]:
    """
    Validate card format and extract BIN information
    Returns: Validation result with tier information
    """
    # Hard constraint: Only masked Visa format
    mask_pattern = r"^4[0-9]{3}-\*{4}-\*{4}-[0-9]{4}$"
    
    if not re.match(mask_pattern, card_number):
        return {
            "valid": False,
            "error": "Invalid card format. Only masked Visa cards accepted.",
            "tier": None
        }
    
    # Reject full PANs
    if not re.search(r"\*{4}", card_number):
        return {
            "valid": False,
            "error": "Full card numbers are not accepted.",
            "tier": None
        }
    
    # Extract BIN
    first_4 = card_number[:4]
    
    # BIN to Tier mapping (mock - in production use real BIN database)
    bin_tier_map = {
        "4000": "Signature",
        "4111": "Classic",
        "4222": "Infinite",
        "4333": "Signature",
        "4444": "Infinite"
    }
    
    detected_tier = bin_tier_map.get(first_4)
    
    return {
        "valid": detected_tier is not None,
        "tier": detected_tier,
        "bin": first_4,
        "error": None if detected_tier else f"Unsupported BIN: {first_4}"
    }


def filter_by_location_tool(benefits: List[Dict], location: str) -> Dict[str, Any]:
    """
    Filter benefits by location with strict matching
    Returns: Filtered benefits list
    """
    location_map = {
        "bangalore": ["bangalore", "bengaluru"],
        "chennai": ["chennai", "madras"],
        "mumbai": ["mumbai", "bombay"],
        "goa": ["goa"]
    }
    
    # Get location variants
    location_variants = []
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
    for key, variants in location_map.items():
        if location_lower in variants or any(v in location_lower for v in variants):
            other_cities_to_check = other_cities.get(key, [])
            break
    
    # Filter benefits
    filtered = []
    for benefit in benefits:
        content = benefit.get("content", "").lower()
        metadata_location = benefit.get("metadata", {}).get("location", "").lower() if benefit.get("metadata") else ""
        
        # Check if matches selected location
        has_selected_location = any(variant in content for variant in location_variants) or \
                               any(variant in metadata_location for variant in location_variants)
        
        # Check if mentions other cities
        has_other_city = any(city in content for city in other_cities_to_check)
        
        # Include only if matches location and doesn't mention other cities
        if has_selected_location and not (has_other_city and not has_selected_location):
            filtered.append(benefit)
    
    return {
        "filtered_benefits": filtered,
        "original_count": len(benefits),
        "filtered_count": len(filtered),
        "location": location
    }


def score_benefits_tool(benefits: List[Dict], user_context: Optional[str] = None, 
                       location: Optional[str] = None) -> Dict[str, Any]:
    """
    Score and rank benefits based on relevance
    Returns: Ranked benefits with scores
    """
    WEIGHTS = {
        "lifestyle": 0.35,
        "location": 0.30,
        "temporal": 0.20,
        "monetary": 0.15
    }
    
    context = (user_context or "").lower()
    location_lower = (location or "").lower()
    
    # Location variants
    location_map = {
        "bangalore": ["bangalore", "bengaluru"],
        "chennai": ["chennai", "madras"],
        "mumbai": ["mumbai", "bombay"],
        "goa": ["goa"]
    }
    
    location_variants = []
    if location_lower:
        for key, variants in location_map.items():
            if location_lower in variants or any(v in location_lower for v in variants):
                location_variants = variants
                break
        if not location_variants:
            location_variants = [location_lower]
    
    # Category keywords
    category_keywords = {
        "student": ["student", "education", "campus", "university", "college"],
        "travel": ["travel", "lounge", "airport", "flight", "hotel"],
        "dining_entertainment_shopping": ["dining", "restaurant", "cafe", "shopping", "mall"],
        "services": ["service", "insurance", "protection", "warranty"]
    }
    
    scored_benefits = []
    
    for benefit in benefits:
        content = benefit.get("content", "").lower()
        metadata = benefit.get("metadata", {}) or {}
        
        # Lifestyle score
        category_score = 0.5
        if context in category_keywords:
            matches = sum(1 for kw in category_keywords[context] if kw in content)
            category_score = min(matches / max(len(category_keywords[context]), 1), 1.0)
        
        # Location score
        location_score = 0.5
        if location_lower and location_variants:
            if any(variant in content for variant in location_variants):
                location_score = 1.0
            else:
                location_score = 0.0
        else:
            location_score = 0.5
        
        # Temporal score
        temporal_score = 0.5
        if "2026" in content or "2027" in content:
            temporal_score = 1.0
        elif "2025" in content:
            temporal_score = 0.7
        elif "expired" in content or "ended" in content:
            temporal_score = 0.1
        
        # Monetary score
        monetary_score = 0.5
        dollar_matches = re.findall(r'\$(\d+)', content)
        if dollar_matches:
            amounts = [int(amt) for amt in dollar_matches if amt.isdigit()]
            if amounts:
                max_amount = max(amounts)
                if max_amount >= 500:
                    monetary_score = 1.0
                elif max_amount >= 100:
                    monetary_score = 0.7
                elif max_amount >= 50:
                    monetary_score = 0.5
        
        # Total score
        total_score = (
            category_score * WEIGHTS["lifestyle"] +
            location_score * WEIGHTS["location"] +
            temporal_score * WEIGHTS["temporal"] +
            monetary_score * WEIGHTS["monetary"]
        )
        
        scored_benefits.append({
            "benefit": benefit,
            "scores": {
                "lifestyle": category_score,
                "location": location_score,
                "temporal": temporal_score,
                "monetary": monetary_score,
                "total": total_score
            }
        })
    
    # Sort by total score
    scored_benefits.sort(key=lambda x: x["scores"]["total"], reverse=True)
    
    return {
        "ranked_benefits": scored_benefits,
        "count": len(scored_benefits)
    }


def check_compliance_tool(text: str) -> Dict[str, Any]:
    """
    Check text for compliance violations
    Returns: Compliance check result
    """
    violations = []
    compliance_approved = True
    
    # Check for unsafe language
    unsafe_patterns = [
        (r"guaranteed", "Unsafe language: 'guaranteed'"),
        (r"always", "Unsafe language: 'always'"),
        (r"never", "Unsafe language: 'never'"),
        (r"100%", "Unsafe language: '100%'"),
        (r"definitely", "Unsafe language: 'definitely'")
    ]
    
    for pattern, message in unsafe_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            violations.append(message)
            compliance_approved = False
    
    # Check for PAN
    if re.search(r"\d{4}-\d{4}-\d{4}-\d{4}", text):
        violations.append("Potential PAN detected in output")
        compliance_approved = False
    
    return {
        "approved": compliance_approved,
        "violations": violations,
        "text_length": len(text)
    }


# ============================================================================
# TOOL REGISTRY
# ============================================================================

TOOL_REGISTRY = {
    "search_benefits": Tool(
        name="search_benefits",
        description="Search for Visa card benefits in the vector database. Parameters: query (str), tier (str), location (str, optional), k (int, default=20)",
        func=search_benefits_tool
    ),
    "validate_card": Tool(
        name="validate_card",
        description="Validate a masked Visa card number and extract tier information. Parameters: card_number (str)",
        func=validate_card_tool
    ),
    "filter_by_location": Tool(
        name="filter_by_location",
        description="Filter benefits by location with strict matching. Parameters: benefits (list), location (str)",
        func=filter_by_location_tool
    ),
    "score_benefits": Tool(
        name="score_benefits",
        description="Score and rank benefits by relevance. Parameters: benefits (list), user_context (str, optional), location (str, optional)",
        func=score_benefits_tool
    ),
    "check_compliance": Tool(
        name="check_compliance",
        description="Check text for compliance violations (unsafe language, PAN detection). Parameters: text (str)",
        func=check_compliance_tool
    )
}


def get_tool(name: str) -> Optional[Tool]:
    """Get a tool by name"""
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[Dict[str, str]]:
    """List all available tools"""
    return [tool.to_dict() for tool in TOOL_REGISTRY.values()]


def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name"""
    tool = get_tool(tool_name)
    if not tool:
        return {
            "success": False,
            "error": f"Tool '{tool_name}' not found",
            "tool": tool_name
        }
    return tool(**kwargs)


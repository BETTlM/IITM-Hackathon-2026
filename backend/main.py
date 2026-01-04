"""
FastAPI Server for Visa Benefits Workflow
Handles user input validation and orchestrates the agent workflow
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import re
import os
from typing import Optional, Literal
from app.graph import app as workflow_app, AgentState

# Auto-run ingestion on startup
def run_ingestion():
    """Automatically ingest benefits data on server startup"""
    try:
        import sys
        import os
        # Add current directory to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import ingest_simple, fallback to ingest
        try:
            from ingest_simple import ingest_data
        except ImportError:
            from ingest import ingest_data
        
        print("🔄 Running automatic data ingestion...")
        ingest_data()
        print("✅ Data ingestion completed successfully!")
    except Exception as e:
        print(f"⚠️  Warning: Data ingestion failed: {e}")
        print("   You can manually run: python ingest_simple.py or python ingest.py")

# Initialize FastAPI
api = FastAPI(
    title="Visa Benefits API",
    description="Awareness-only system for Visa card benefits using local LLMs",
    version="1.0.0"
)

# CORS middleware - configure allowed origins
import os
cors_origins = os.getenv("CORS_ORIGINS", "*")
if cors_origins != "*":
    # Parse comma-separated origins
    allowed_origins = [origin.strip() for origin in cors_origins.split(",")]
else:
    allowed_origins = ["*"]

api.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INPUT VALIDATION MODELS
# ============================================================================

class BenefitRequest(BaseModel):
    """User input structure"""
    card_number: str = Field(
        ...,
        description="Masked Visa card number (format: 4XXX-****-****-XXXX)",
        example="4111-****-****-1111"
    )
    user_context: Optional[str] = Field(
        None,
        description="Lifestyle context: student, traveler, or family",
        example="student"
    )
    preferred_language: Literal["en", "ta"] = Field(
        "en",
        description="Preferred language: en (English) or ta (Tamil)"
    )
    location: Optional[str] = Field(
        None,
        description="Coarse location (city-level only)",
        example="Chennai"
    )
    
    @validator("card_number")
    def validate_card_format(cls, v):
        """HARD CONSTRAINT: Only masked Visa cards"""
        pattern = r"^4[0-9]{3}-\*{4}-\*{4}-[0-9]{4}$"
        if not re.match(pattern, v):
            raise ValueError(
                "Invalid card format. Only masked Visa cards accepted. "
                "Format: 4XXX-****-****-XXXX (e.g., 4111-****-****-1111)"
            )
        
        # Reject full PANs
        if not re.search(r"\*{4}", v):
            raise ValueError("Full card numbers are not accepted. Use masked format.")
        
        return v
    
    @validator("user_context")
    def validate_context(cls, v):
        """Validate user context"""
        valid_contexts = ["student", "travel", "dining_entertainment_shopping", "services"]
        if v and v.lower() not in valid_contexts:
            raise ValueError(f"user_context must be one of: {', '.join(valid_contexts)}")
        return v.lower() if v else None


class BenefitResponse(BaseModel):
    """Response structure"""
    status: str
    card_tier: Optional[str] = None
    recommended_benefit: Optional[dict] = None
    recommendations: Optional[list] = None  # Top 4 recommendations
    all_benefits: Optional[list] = None  # All eligible suggestions
    total_benefits_count: Optional[int] = None
    disclaimers: Optional[list] = None
    language: Optional[str] = None
    metadata: Optional[dict] = None
    error_code: Optional[str] = None
    message: Optional[str] = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@api.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Visa Benefits API",
        "version": "1.0.0",
        "constraints": [
            "Accepts only masked Visa card numbers",
            "No data persistence",
            "Awareness-only (no transactions)",
            "All benefits RAG-grounded",
            "Local LLM processing"
        ]
    }


@api.post("/benefits", response_model=BenefitResponse)
async def get_benefits(request: BenefitRequest):
    """
    Main endpoint: Get personalized Visa card benefits
    
    Workflow:
    1. Supervisor Agent - Orchestrates flow
    2. Card Intelligence Agent - Validates card, determines tier
    3. Benefit Retrieval Agent - RAG-based benefit search
    4. Explanation Agent - Converts to plain language
    5. Recommendation Agent - Ranks benefits by relevance
    6. Language Agent - Translates if needed
    7. Compliance Agent - Final validation and disclaimers
    """
    try:
        # Initialize state
        initial_state: AgentState = {
            "card_number": request.card_number,
            "user_context": request.user_context,
            "preferred_language": request.preferred_language,
            "location": request.location,
            "detected_tier": None,
            "bin_valid": False,
            "retrieved_docs": [],
            "benefit_chunks": [],
            "plain_language_explanation": None,
            "ranked_benefits": [],
            "top_benefit": None,
            "translated_response": None,
            "compliance_approved": False,
            "disclaimers": [],
            "final_output": None,
            "error": None,
            "error_code": None
        }
        
        # Execute workflow
        result = workflow_app.invoke(initial_state)
        
        # Extract final output
        final_output = result.get("final_output")
        
        if not final_output:
            raise HTTPException(
                status_code=500,
                detail="Workflow execution failed - no output generated"
            )
        
        # Return response
        if final_output.get("status") == "error":
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": final_output.get("error_code"),
                    "message": final_output.get("message")
                }
            )
        
        return BenefitResponse(**final_output)
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@api.get("/health")
def health_check():
    """Detailed health check"""
    try:
        # Test workflow compilation
        test_state: AgentState = {
            "card_number": "4111-****-****-1111",
            "user_context": None,
            "preferred_language": "en",
            "location": None,
            "detected_tier": None,
            "bin_valid": False,
            "retrieved_docs": [],
            "benefit_chunks": [],
            "plain_language_explanation": None,
            "ranked_benefits": [],
            "top_benefit": None,
            "translated_response": None,
            "compliance_approved": False,
            "disclaimers": [],
            "final_output": None,
            "error": None,
            "error_code": None
        }
        
        return {
            "status": "healthy",
            "workflow": "compiled",
            "constraints": {
                "masked_cards_only": True,
                "no_persistence": True,
                "awareness_only": True,
                "rag_grounded": True,
                "local_llm": True
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    # Run ingestion on startup
    run_ingestion()
    print("🚀 Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(api, host="0.0.0.0", port=8000)


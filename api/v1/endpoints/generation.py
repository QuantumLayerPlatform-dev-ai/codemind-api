"""
Code Generation API endpoints
"""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

try:
    from ....services.llm_router import LLMRouter, LLMRequest, TaskType
    from ....core.logging import get_logger
except ImportError:
    from services.llm_router import LLMRouter, LLMRequest, TaskType
    from core.logging import get_logger

logger = get_logger("generation_api")

router = APIRouter()


class GenerationRequest(BaseModel):
    """Request model for application generation"""
    business_description: str = Field(..., description="Business description or idea")
    complexity: float = Field(default=0.5, ge=0.0, le=1.0, description="Task complexity (0.0-1.0)")
    requirements: Dict[str, Any] = Field(default_factory=dict, description="Additional requirements")


class GenerationResponse(BaseModel):
    """Response model for application generation"""
    request_id: str
    status: str
    message: str
    data: Dict[str, Any] = Field(default_factory=dict)


@router.post("/app", response_model=GenerationResponse)
async def generate_application(
    request: GenerationRequest,
    bg_tasks: BackgroundTasks,
    api_request: Request
) -> GenerationResponse:
    """Generate a complete application from business description"""

    try:
        logger.info(f"🚀 Starting application generation for: {request.business_description[:100]}...")

        # Create LLM request for business intent analysis
        llm_request = LLMRequest(
            prompt=f"""Analyze this business idea and extract key requirements:

Business Description: {request.business_description}

Please provide:
1. Industry classification
2. Core business model
3. Key features needed
4. Technical requirements
5. Compliance considerations

Additional Requirements: {request.requirements}""",
            task_type=TaskType.BUSINESS_INTENT,
            complexity=request.complexity,
            max_tokens=2048
        )

        # Get LLM router from app state
        llm_router: LLMRouter = api_request.app.state.llm_router

        # Process the request
        llm_response = await llm_router.route_request(llm_request)

        logger.info(f"✅ Business intent analyzed for request {llm_response.request_id}")

        return GenerationResponse(
            request_id=llm_response.request_id,
            status="success",
            message="Business intent analyzed successfully",
            data={
                "business_analysis": llm_response.content,
                "model_used": llm_response.model_used,
                "tokens_used": llm_response.tokens_used,
                "cost_usd": llm_response.cost_usd,
                "duration_ms": llm_response.duration_ms,
                "confidence": llm_response.confidence
            }
        )

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )


@router.post("/code", response_model=GenerationResponse)
async def generate_code(
    request: GenerationRequest,
    api_request: Request
) -> GenerationResponse:
    """Generate code for a specific component or feature"""

    try:
        logger.info(f"💻 Starting code generation for: {request.business_description[:100]}...")

        # Create LLM request for code generation
        llm_request = LLMRequest(
            prompt=f"""Generate production-ready code for this requirement:

Requirement: {request.business_description}

Please provide:
1. Clean, well-documented code
2. Error handling
3. Input validation
4. Basic tests
5. Usage examples

Additional Context: {request.requirements}""",
            task_type=TaskType.CODE_GENERATION,
            complexity=request.complexity,
            max_tokens=4096
        )

        # Get LLM router from app state
        llm_router: LLMRouter = api_request.app.state.llm_router

        # Process the request
        llm_response = await llm_router.route_request(llm_request)

        logger.info(f"✅ Code generated for request {llm_response.request_id}")

        return GenerationResponse(
            request_id=llm_response.request_id,
            status="success",
            message="Code generated successfully",
            data={
                "generated_code": llm_response.content,
                "model_used": llm_response.model_used,
                "tokens_used": llm_response.tokens_used,
                "cost_usd": llm_response.cost_usd,
                "duration_ms": llm_response.duration_ms,
                "confidence": llm_response.confidence
            }
        )

    except Exception as e:
        logger.error(f"❌ Code generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Code generation failed: {str(e)}"
        )


@router.get("/status/{request_id}")
async def get_generation_status(request_id: str) -> GenerationResponse:
    """Get the status of a generation request"""

    # TODO: Implement status tracking
    return GenerationResponse(
        request_id=request_id,
        status="completed",
        message="Generation completed",
        data={"progress": 100}
    )
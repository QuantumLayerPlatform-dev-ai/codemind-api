"""
LLM Router Service for CodeMind

Routes requests to the appropriate LLM based on task requirements,
cost optimization, and performance characteristics.
"""

import asyncio
import hashlib
import json
import time
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

import boto3
from openai import AsyncAzureOpenAI
import httpx

try:
    from ..core.logging import get_logger
    from ..core.config import get_settings
except ImportError:
    from core.logging import get_logger
    from core.config import get_settings

logger = get_logger("llm_router")
settings = get_settings()


class ModelType(str, Enum):
    """Available model types"""
    CLAUDE_3_7_SONNET = "claude-3-7-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    GPT_5 = "gpt-5"
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    O4_MINI = "o4-mini"
    GPT_3_5_TURBO = "gpt-3.5-turbo"


class TaskType(str, Enum):
    """Types of tasks for model routing"""
    BUSINESS_INTENT = "business_intent"
    CODE_GENERATION = "code_generation"
    ARCHITECTURE_DESIGN = "architecture_design"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    FORMATTING = "formatting"
    SIMPLE_COMPLETION = "simple_completion"


@dataclass
class LLMRequest:
    """LLM request structure"""
    prompt: str
    task_type: TaskType
    complexity: float = 0.5  # 0.0 - 1.0
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    context: Dict[str, Any] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    model_used: str
    tokens_used: Dict[str, int]
    cost_usd: float
    duration_ms: int
    confidence: float
    request_id: str


class LLMRouter:
    """Smart LLM routing service"""

    def __init__(self):
        self.settings = settings
        self.model_costs = {
            ModelType.CLAUDE_3_7_SONNET: {"input": 0.003, "output": 0.015},
            ModelType.CLAUDE_3_HAIKU: {"input": 0.00025, "output": 0.00125},
            ModelType.CLAUDE_3_SONNET: {"input": 0.003, "output": 0.015},
            ModelType.GPT_5: {"input": 0.05, "output": 0.10},
            ModelType.GPT_4_1: {"input": 0.03, "output": 0.06},
            ModelType.GPT_4_1_MINI: {"input": 0.015, "output": 0.03},
            ModelType.GPT_4_1_NANO: {"input": 0.005, "output": 0.01},
            ModelType.O4_MINI: {"input": 0.008, "output": 0.016},
            ModelType.GPT_3_5_TURBO: {"input": 0.0005, "output": 0.0015},
        }

        # Routing rules
        self.routing_rules = {
            TaskType.BUSINESS_INTENT: [ModelType.GPT_5, ModelType.CLAUDE_3_7_SONNET],
            TaskType.CODE_GENERATION: [ModelType.CLAUDE_3_7_SONNET, ModelType.GPT_4_1],
            TaskType.ARCHITECTURE_DESIGN: [ModelType.GPT_5, ModelType.CLAUDE_3_7_SONNET],
            TaskType.TESTING: [ModelType.GPT_4_1_MINI, ModelType.O4_MINI],
            TaskType.DOCUMENTATION: [ModelType.GPT_3_5_TURBO, ModelType.GPT_4_1_NANO],
            TaskType.FORMATTING: [ModelType.GPT_4_1_NANO, ModelType.CLAUDE_3_HAIKU],
            TaskType.SIMPLE_COMPLETION: [ModelType.GPT_4_1_NANO, ModelType.CLAUDE_3_HAIKU],
        }

        # Initialize clients
        self.bedrock_client = None
        self.azure_client = None

    async def initialize(self):
        """Initialize LLM clients"""
        try:
            # Initialize AWS Bedrock
            if self.settings.aws_access_key_id:
                self.bedrock_client = boto3.client(
                    'bedrock-runtime',
                    region_name=self.settings.aws_region,
                    aws_access_key_id=self.settings.aws_access_key_id,
                    aws_secret_access_key=self.settings.aws_secret_access_key
                )
                logger.info("✅ AWS Bedrock client initialized")
            else:
                logger.warning("⚠️ AWS credentials not provided, Bedrock disabled")

            # Initialize Azure OpenAI
            if self.settings.azure_openai_endpoint and self.settings.azure_openai_api_key:
                self.azure_client = AsyncAzureOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    api_version=self.settings.azure_openai_api_version,
                    azure_endpoint=self.settings.azure_openai_endpoint
                )
                logger.info("✅ Azure OpenAI client initialized")
            else:
                logger.warning("⚠️ Azure OpenAI credentials not provided, Azure models disabled")

        except Exception as e:
            logger.error(f"❌ Failed to initialize LLM clients: {e}")
            raise

    async def route_request(self, request: LLMRequest) -> LLMResponse:
        """Route request to optimal model"""
        start_time = time.time()
        request_id = hashlib.sha256(
            f"{request.prompt[:100]}{request.task_type}{time.time()}".encode()
        ).hexdigest()[:16]

        try:
            # Select optimal model
            model = self.select_model(request)
            logger.info(f"🎯 Routing request {request_id} to {model}")

            # Generate response
            if model in [ModelType.CLAUDE_3_7_SONNET, ModelType.CLAUDE_3_HAIKU, ModelType.CLAUDE_3_SONNET]:
                response = await self._call_bedrock(model, request)
            else:
                response = await self._call_azure(model, request)

            # Calculate metrics
            duration_ms = int((time.time() - start_time) * 1000)
            cost = self.calculate_cost(model, response["tokens_used"])

            return LLMResponse(
                content=response["content"],
                model_used=model.value,
                tokens_used=response["tokens_used"],
                cost_usd=cost,
                duration_ms=duration_ms,
                confidence=response.get("confidence", 0.8),
                request_id=request_id
            )

        except Exception as e:
            logger.error(f"❌ Failed to process request {request_id}: {e}")
            raise

    def select_model(self, request: LLMRequest) -> ModelType:
        """Select optimal model based on task and complexity"""
        candidate_models = self.routing_rules.get(request.task_type, [ModelType.GPT_4_1])

        # Filter to only AWS Bedrock models if Azure is not available
        if not self.azure_client:
            candidate_models = [m for m in candidate_models if m.value.startswith('claude')]

        # Fallback to Claude if no candidates
        if not candidate_models:
            candidate_models = [ModelType.CLAUDE_3_HAIKU]

        # For high complexity tasks, use premium models
        if request.complexity > 0.8:
            return candidate_models[0]  # Best model

        # For medium complexity, use balanced model
        elif request.complexity > 0.4:
            return candidate_models[min(1, len(candidate_models) - 1)]

        # For low complexity, use cheapest capable model
        else:
            return candidate_models[-1] if candidate_models else ModelType.CLAUDE_3_HAIKU

    async def _call_bedrock(self, model: ModelType, request: LLMRequest) -> Dict[str, Any]:
        """Call AWS Bedrock Claude models"""
        if not self.bedrock_client:
            raise ValueError("Bedrock client not initialized")

        # Map model to Bedrock model ID (actual available models in eu-west-2)
        model_mapping = {
            ModelType.CLAUDE_3_7_SONNET: "anthropic.claude-3-7-sonnet-20250219-v1:0",
            ModelType.CLAUDE_3_HAIKU: "anthropic.claude-3-haiku-20240307-v1:0",
            ModelType.CLAUDE_3_SONNET: "anthropic.claude-3-sonnet-20240229-v1:0",
        }

        model_id = model_mapping.get(model)
        if not model_id:
            raise ValueError(f"Unsupported Bedrock model: {model}")

        # Prepare request
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ]
        }

        try:
            response = self.bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body)
            )

            response_body = json.loads(response['body'].read())

            return {
                "content": response_body['content'][0]['text'],
                "tokens_used": {
                    "input": response_body.get('usage', {}).get('input_tokens', 0),
                    "output": response_body.get('usage', {}).get('output_tokens', 0)
                },
                "confidence": 0.9  # Claude typically high confidence
            }

        except Exception as e:
            logger.error(f"Bedrock API error: {e}")
            raise

    async def _call_azure(self, model: ModelType, request: LLMRequest) -> Dict[str, Any]:
        """Call Azure OpenAI models"""
        if not self.azure_client:
            raise ValueError("Azure OpenAI client not initialized")

        # Map model to Azure deployment name (based on your Azure OpenAI deployments)
        model_mapping = {
            ModelType.GPT_5: "gpt-5",                    # Your deployment name
            ModelType.GPT_4_1: "gpt-4.1",                # Your deployment name
            ModelType.GPT_4_1_MINI: "gpt-4.1-mini",      # Your deployment name
            ModelType.GPT_4_1_NANO: "gpt-4.1-nano",      # Your deployment name
            ModelType.O4_MINI: "o4-mini",                # Your deployment name
            ModelType.GPT_3_5_TURBO: "gpt-35-turbo",     # Fallback if available
        }

        deployment_name = model_mapping.get(model)
        if not deployment_name:
            raise ValueError(f"Unsupported Azure model: {model}")

        try:
            response = await self.azure_client.chat.completions.create(
                model=deployment_name,
                messages=[
                    {
                        "role": "user",
                        "content": request.prompt
                    }
                ],
                max_tokens=request.max_tokens or 4096,
                temperature=request.temperature
            )

            return {
                "content": response.choices[0].message.content,
                "tokens_used": {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens
                },
                "confidence": 0.85  # Default confidence for GPT models
            }

        except Exception as e:
            logger.error(f"Azure OpenAI API error: {e}")
            raise

    def calculate_cost(self, model: ModelType, tokens_used: Dict[str, int]) -> float:
        """Calculate cost for the request"""
        costs = self.model_costs.get(model, {"input": 0.01, "output": 0.02})

        input_cost = (tokens_used.get("input", 0) / 1000) * costs["input"]
        output_cost = (tokens_used.get("output", 0) / 1000) * costs["output"]

        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check if LLM services are healthy"""
        try:
            # Quick test with minimal request
            test_request = LLMRequest(
                prompt="Hello",
                task_type=TaskType.SIMPLE_COMPLETION,
                complexity=0.1,
                max_tokens=10
            )

            # Try with cheapest model
            model = ModelType.CLAUDE_3_HAIKU if self.bedrock_client else ModelType.GPT_4_1_NANO

            if model.value.startswith("claude") and self.bedrock_client:
                await self._call_bedrock(model, test_request)
            elif self.azure_client:
                await self._call_azure(model, test_request)
            else:
                return False

            return True

        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
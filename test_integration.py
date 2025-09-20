#!/usr/bin/env python3
"""
Integration test script for CodeMind API
Tests LLM integration and basic functionality
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup basic logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a simplified config for testing
class TestSettings:
    def __init__(self):
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION', 'eu-west-2')
        self.azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.azure_openai_api_key = os.getenv('AZURE_OPENAI_API_KEY')
        self.azure_openai_api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')

# Simple logger
logger = logging.getLogger("integration_test")

# Setup logging
setup_logging()
logger = get_logger("integration_test")

async def test_llm_integration():
    """Test LLM router integration"""
    logger.info("🧪 Starting LLM integration test")

    settings = get_settings()

    # Check if credentials are available
    logger.info("📋 Checking credentials...")

    aws_available = bool(settings.aws_access_key_id and settings.aws_secret_access_key)
    azure_available = bool(settings.azure_openai_endpoint and settings.azure_openai_api_key)

    logger.info(f"AWS Bedrock available: {aws_available}")
    logger.info(f"Azure OpenAI available: {azure_available}")

    if not aws_available and not azure_available:
        logger.warning("⚠️ No LLM credentials configured. Please set environment variables:")
        logger.warning("AWS: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        logger.warning("Azure: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")
        return False

    # Initialize LLM router
    try:
        llm_router = LLMRouter()
        await llm_router.initialize()
        logger.info("✅ LLM Router initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize LLM Router: {e}")
        return False

    # Test simple completion
    try:
        logger.info("🎯 Testing simple completion...")

        test_request = LLMRequest(
            prompt="Say 'Hello CodeMind!' and explain in one sentence what you are.",
            task_type=TaskType.SIMPLE_COMPLETION,
            complexity=0.1,
            max_tokens=100
        )

        response = await llm_router.route_request(test_request)

        logger.info(f"✅ Response received:")
        logger.info(f"  Model: {response.model_used}")
        logger.info(f"  Content: {response.content[:100]}...")
        logger.info(f"  Tokens: {response.tokens_used}")
        logger.info(f"  Cost: ${response.cost_usd:.4f}")
        logger.info(f"  Duration: {response.duration_ms}ms")

        return True

    except Exception as e:
        logger.error(f"❌ LLM test failed: {e}")
        return False

async def test_business_intent():
    """Test business intent analysis"""
    logger.info("💼 Testing business intent analysis...")

    try:
        llm_router = LLMRouter()
        await llm_router.initialize()

        test_request = LLMRequest(
            prompt="I want to build a task management app for small teams with real-time collaboration features.",
            task_type=TaskType.BUSINESS_INTENT,
            complexity=0.6,
            max_tokens=500
        )

        response = await llm_router.route_request(test_request)

        logger.info(f"✅ Business intent analyzed:")
        logger.info(f"  Model: {response.model_used}")
        logger.info(f"  Analysis: {response.content[:200]}...")
        logger.info(f"  Cost: ${response.cost_usd:.4f}")

        return True

    except Exception as e:
        logger.error(f"❌ Business intent test failed: {e}")
        return False

async def test_health_check():
    """Test LLM health check"""
    logger.info("🏥 Testing health check...")

    try:
        llm_router = LLMRouter()
        await llm_router.initialize()

        is_healthy = await llm_router.health_check()

        if is_healthy:
            logger.info("✅ LLM services are healthy")
            return True
        else:
            logger.error("❌ LLM health check failed")
            return False

    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    logger.info("🚀 Starting CodeMind integration tests")

    tests = [
        ("LLM Integration", test_llm_integration),
        ("Business Intent", test_business_intent),
        ("Health Check", test_health_check),
    ]

    results = []

    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")

        try:
            result = await test_func()
            results.append((test_name, result))

            if result:
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")

        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")
            results.append((test_name, False))

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")

    logger.info(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! CodeMind is ready.")
        return True
    else:
        logger.error("💥 Some tests failed. Please check configuration.")
        return False

if __name__ == "__main__":
    # Check for environment file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️ No .env file found. Creating template...")

        env_template = """# CodeMind API Configuration
# Copy this file to .env and fill in your credentials

# Database
DATABASE_URL=postgresql+asyncpg://postgres:codemind-dev-password@localhost:5432/codemind

# Redis
REDIS_URL=redis://:codemind-dev-password@localhost:6379

# AWS Bedrock (Claude models)
AWS_REGION=eu-west-2
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here

# Azure OpenAI (GPT models)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
AZURE_OPENAI_API_VERSION=2024-02-01

# Security
SECRET_KEY=your-secret-key-change-in-production

# Other services
QDRANT_URL=http://localhost:6333
NATS_URL=nats://localhost:4222
TEMPORAL_HOST=localhost
TEMPORAL_PORT=7233

# Debug
DEBUG=true
LOG_LEVEL=INFO
"""

        with open(env_file, "w") as f:
            f.write(env_template)

        print(f"📝 Template .env file created at {env_file}")
        print("Please edit .env with your actual credentials and run again.")
        sys.exit(1)

    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
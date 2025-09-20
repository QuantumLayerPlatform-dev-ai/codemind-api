#!/usr/bin/env python3
"""
Simple integration test for CodeMind - Tests core functionality without relative imports
"""

import asyncio
import os
import json
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simple_test")

class TaskType(str, Enum):
    """Types of tasks for model routing"""
    BUSINESS_INTENT = "business_intent"
    CODE_GENERATION = "code_generation"
    SIMPLE_COMPLETION = "simple_completion"

@dataclass
class LLMRequest:
    """LLM request structure"""
    prompt: str
    task_type: TaskType
    complexity: float = 0.5
    max_tokens: Optional[int] = None
    temperature: float = 0.7

async def test_aws_bedrock():
    """Test AWS Bedrock connectivity"""
    logger.info("🧪 Testing AWS Bedrock connectivity...")

    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION', 'eu-west-2')

    if not aws_access_key or not aws_secret_key:
        logger.warning("⚠️ AWS credentials not found in environment")
        return False

    try:
        import boto3

        bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        # Test with Claude 3 Haiku (cheapest option)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 50,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello CodeMind!' and nothing else."
                }
            ]
        }

        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(body)
        )

        response_body = json.loads(response['body'].read())
        content = response_body['content'][0]['text']

        logger.info(f"✅ AWS Bedrock test successful!")
        logger.info(f"  Model: Claude 3 Haiku")
        logger.info(f"  Response: {content}")
        logger.info(f"  Tokens: {response_body.get('usage', {})}")

        return True

    except Exception as e:
        logger.error(f"❌ AWS Bedrock test failed: {e}")
        return False

async def test_azure_openai():
    """Test Azure OpenAI connectivity"""
    logger.info("🧪 Testing Azure OpenAI connectivity...")

    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-01')

    if not endpoint or not api_key:
        logger.warning("⚠️ Azure OpenAI credentials not found in environment")
        return False

    try:
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        # Test with GPT-4.1-nano (cheapest option)
        response = await client.chat.completions.create(
            model="gpt-4.1-nano",  # Assuming this is your deployment name
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello CodeMind!' and nothing else."
                }
            ],
            max_tokens=50,
            temperature=0.7
        )

        content = response.choices[0].message.content
        tokens = response.usage

        logger.info(f"✅ Azure OpenAI test successful!")
        logger.info(f"  Model: GPT-4.1-nano")
        logger.info(f"  Response: {content}")
        logger.info(f"  Tokens: {tokens}")

        return True

    except Exception as e:
        logger.error(f"❌ Azure OpenAI test failed: {e}")
        return False

async def test_infrastructure():
    """Test infrastructure connectivity"""
    logger.info("🧪 Testing infrastructure connectivity...")

    tests = []

    # Test PostgreSQL
    try:
        import asyncpg
        conn = await asyncpg.connect(
            "postgresql://postgres:codemind-dev-password@192.168.1.177:30432/codemind"
        )
        await conn.execute("SELECT 1")
        await conn.close()
        logger.info("✅ PostgreSQL connection successful")
        tests.append(("PostgreSQL", True))
    except Exception as e:
        logger.error(f"❌ PostgreSQL connection failed: {e}")
        tests.append(("PostgreSQL", False))

    # Test Redis
    try:
        import redis
        r = redis.Redis.from_url("redis://:codemind-dev-password@192.168.1.177:30379")
        r.ping()
        logger.info("✅ Redis connection successful")
        tests.append(("Redis", True))
    except Exception as e:
        logger.error(f"❌ Redis connection failed: {e}")
        tests.append(("Redis", False))

    # Test Qdrant
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get("http://192.168.1.177:30333/")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Qdrant connection successful - Version: {data.get('version')}")
                tests.append(("Qdrant", True))
            else:
                logger.error(f"❌ Qdrant health check failed: {response.status_code}")
                tests.append(("Qdrant", False))
    except Exception as e:
        logger.error(f"❌ Qdrant connection failed: {e}")
        tests.append(("Qdrant", False))

    # Test NATS
    try:
        import nats
        nc = await nats.connect("nats://192.168.1.177:30422")
        await nc.close()
        logger.info("✅ NATS connection successful")
        tests.append(("NATS", True))
    except Exception as e:
        logger.error(f"❌ NATS connection failed: {e}")
        tests.append(("NATS", False))

    return tests

async def run_all_tests():
    """Run all tests"""
    logger.info("🚀 Starting CodeMind Simple Integration Tests")

    # Check for .env file
    env_file = ".env"
    if not os.path.exists(env_file):
        logger.warning("⚠️ No .env file found. Please create one with your credentials.")
        logger.info("Expected environment variables:")
        logger.info("  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")
        logger.info("  AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY")

    results = []

    # Test infrastructure
    logger.info(f"\n{'='*50}")
    logger.info("Testing Infrastructure")
    logger.info(f"{'='*50}")

    infra_tests = await test_infrastructure()
    results.extend(infra_tests)

    # Test LLM providers
    logger.info(f"\n{'='*50}")
    logger.info("Testing LLM Providers")
    logger.info(f"{'='*50}")

    aws_result = await test_aws_bedrock()
    results.append(("AWS Bedrock", aws_result))

    azure_result = await test_azure_openai()
    results.append(("Azure OpenAI", azure_result))

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

    if passed >= 3:  # At least basic infrastructure working
        logger.info("🎉 Core systems are operational!")
        return True
    else:
        logger.error("💥 Critical systems failed. Check your setup.")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
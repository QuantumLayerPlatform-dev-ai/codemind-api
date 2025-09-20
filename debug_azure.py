#!/usr/bin/env python3
"""
Debug Azure OpenAI connection issues
"""

import os
import asyncio
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("azure_debug")

async def debug_azure_openai():
    """Debug Azure OpenAI connection"""

    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    api_version = os.getenv('AZURE_OPENAI_API_VERSION')

    logger.info("🔍 Azure OpenAI Configuration:")
    logger.info(f"  Endpoint: {endpoint}")
    logger.info(f"  API Key: {api_key[:10]}...{api_key[-4:] if api_key else 'None'}")
    logger.info(f"  API Version: {api_version}")

    if not endpoint or not api_key:
        logger.error("❌ Missing Azure credentials")
        return False

    try:
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )

        # Test different deployment names
        deployment_names = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1", "gpt-5", "o4-mini"]

        for deployment in deployment_names:
            try:
                logger.info(f"🧪 Testing deployment: {deployment}")

                response = await client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )

                logger.info(f"✅ {deployment}: SUCCESS")
                logger.info(f"  Response: {response.choices[0].message.content}")
                return True

            except Exception as e:
                logger.warning(f"❌ {deployment}: {str(e)[:100]}")
                continue

        logger.error("❌ All deployments failed")
        return False

    except Exception as e:
        logger.error(f"❌ Azure OpenAI client error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_azure_openai())
    exit(0 if success else 1)
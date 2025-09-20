#!/usr/bin/env python3
"""
CodeMind API Server Launcher
Handles imports and starts the FastAPI server
"""

import sys
import os
from pathlib import Path

# Add the api directory to Python path
api_dir = Path(__file__).parent
sys.path.insert(0, str(api_dir))

# Set up environment
os.environ.setdefault('PYTHONPATH', str(api_dir))

# Import and run
if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Import the app
    from main import app

    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production-like behavior
        log_level="info"
    )
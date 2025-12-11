#!/usr/bin/env python3
"""Run the MemoryForge API server."""

import uvicorn
from dotenv import load_dotenv

load_dotenv()


def main():
    """Run the API server."""
    uvicorn.run(
        "memoryforge.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()

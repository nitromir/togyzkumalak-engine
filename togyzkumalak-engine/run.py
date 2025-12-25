#!/usr/bin/env python3
"""
Togyzkumalak Engine - Entry Point

Starts the FastAPI server and serves the frontend.
"""

import os
import sys
import webbrowser
from threading import Timer

# Add parent directory to path for gym_togyzkumalak imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def open_browser():
    """Open browser after server starts."""
    webbrowser.open("http://localhost:8000")


def main():
    """Run the server."""
    import uvicorn
    
    # Open browser after 1.5 seconds
    Timer(1.5, open_browser).start()
    
    print("=" * 60)
    print("  TOGYZKUMALAK ENGINE")
    print("  AI-Powered Toguz Kumalak Game")
    print("=" * 60)
    print()
    print("  Starting server at http://localhost:8000")
    print("  Press Ctrl+C to stop")
    print()
    
    # Check for Gemini API key
    if os.environ.get("GEMINI_API_KEY"):
        print("  Gemini API: Configured")
    else:
        print("  Gemini API: Not configured (set GEMINI_API_KEY for analysis)")
    print()
    print("=" * 60)
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()


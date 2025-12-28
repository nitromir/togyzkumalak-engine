#!/usr/bin/env python3
"""
Togyzkumalak Engine - Entry Point

Starts the FastAPI server and serves the frontend.
Supports deployment via Docker, Vast.ai, and local development.
"""

import os
import sys
import webbrowser
from threading import Timer

# Add parent directory to path for gym_togyzkumalak imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def open_browser(url: str):
    """Open browser after server starts."""
    webbrowser.open(url)


def is_docker():
    """Check if running inside Docker container."""
    return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER', False)


def is_remote():
    """Check if running on remote server (no GUI available)."""
    return (
        is_docker() or
        os.environ.get('SSH_CONNECTION') or
        os.environ.get('VAST_CONTAINERLABEL') or
        os.environ.get('REMOTE_SERVER', False) or
        not os.environ.get('DISPLAY')
    )


def main():
    """Run the server."""
    import uvicorn
    
    # Configuration from environment variables
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8000))
    reload_enabled = os.environ.get('DEV_MODE', 'false').lower() == 'true'
    log_level = os.environ.get('LOG_LEVEL', 'info')
    
    # Only open browser on local machines with GUI
    if not is_remote():
        Timer(1.5, lambda: open_browser(f"http://localhost:{port}")).start()
    
    print("=" * 60)
    print("  TOGYZKUMALAK ENGINE")
    print("  AI-Powered Toguz Kumalak Game")
    print("=" * 60)
    print()
    print(f"  Server URL: http://{host}:{port}")
    print(f"  Environment: {'Docker' if is_docker() else 'Remote' if is_remote() else 'Local'}")
    print("  Press Ctrl+C to stop")
    print()
    
    # Check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print(f"  CUDA: Available ({gpu_count}x {gpu_name})")
        else:
            print("  CUDA: Not available (CPU mode)")
    except ImportError:
        print("  CUDA: PyTorch not available")
    
    # Check for Gemini API key
    if os.environ.get("GEMINI_API_KEY"):
        print("  Gemini API: Configured")
    else:
        print("  Gemini API: Not configured (set GEMINI_API_KEY for analysis)")
    print()
    print("=" * 60)
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload_enabled,
        log_level=log_level,
        workers=1  # Single worker for state consistency
    )


if __name__ == "__main__":
    main()


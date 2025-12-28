#!/usr/bin/env python3
"""
Restart Togyzkumalak server in Jupyter Notebook
Run this in a Jupyter cell to restart the server with latest code
"""

import os
import sys
import subprocess
import time
import requests
import signal

# Get the Python executable path
python_exe = sys.executable

# Get server directory
server_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(server_dir)

print("=" * 60)
print("  RESTARTING TOGYZKUMALAK SERVER")
print("=" * 60)
print()

# Step 1: Check if server is running
try:
    response = requests.get('http://localhost:8000/api/health', timeout=2)
    print("✅ Server is currently running")
    
    # Try to stop it gracefully
    print("🛑 Stopping server...")
    try:
        # Find the process
        result = subprocess.run(
            ['pgrep', '-f', 'python.*run.py'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        print(f"   Sent SIGTERM to process {pid}")
                    except:
                        pass
    except:
        pass
    
    # Wait a bit
    time.sleep(3)
    
except:
    print("ℹ️  Server is not running (or not responding)")
    print()

# Step 2: Pull latest code from GitHub
print("📥 Pulling latest code from GitHub...")
try:
    repo_root = os.path.dirname(server_dir)
    os.chdir(repo_root)
    
    result = subprocess.run(
        ['git', 'pull', 'origin', 'master'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        if "Already up to date" in result.stdout:
            print("✅ Code is already up to date")
        else:
            print("✅ Code updated successfully")
            print(f"   {result.stdout[:200]}...")
    else:
        print(f"⚠️  Git pull had issues: {result.stderr[:200]}")
    
except Exception as e:
    print(f"⚠️  Could not pull from GitHub: {e}")
    print("   Continuing with restart anyway...")

print()

# Step 3: Start server
print("🚀 Starting server...")
os.chdir(server_dir)

# Kill any existing processes first
try:
    subprocess.run(['pkill', '-f', 'python.*run.py'], 
                  capture_output=True, timeout=5)
    time.sleep(2)
except:
    pass

# Start new server process
try:
    # Use nohup to run in background
    process = subprocess.Popen(
        [python_exe, 'run.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=server_dir,
        env=os.environ.copy()
    )
    
    print(f"✅ Server process started (PID: {process.pid})")
    print()
    
    # Wait and check if server is responding
    print("⏳ Waiting for server to start...")
    for i in range(10):
        time.sleep(1)
        try:
            response = requests.get('http://localhost:8000/api/health', timeout=2)
            if response.status_code == 200:
                print("✅ Server is running and responding!")
                print()
                print("=" * 60)
                print("  SERVER RESTARTED SUCCESSFULLY")
                print("=" * 60)
                print()
                print(f"  Server URL: http://localhost:8000")
                print(f"  Process ID: {process.pid}")
                print()
                print("  💡 TIP: If you're accessing via SSH tunnel,")
                print("     refresh your browser at http://localhost:8000")
                print()
                break
        except:
            if i < 9:
                print(f"   ... ({i+1}/10)")
            else:
                print("⚠️  Server started but not responding yet")
                print("   Check the output above for errors")
                print(f"   Process ID: {process.pid}")
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    print()
    print("Try running manually:")
    print(f"  cd {server_dir}")
    print(f"  {python_exe} run.py")

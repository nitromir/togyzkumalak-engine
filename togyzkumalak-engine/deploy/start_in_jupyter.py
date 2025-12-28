#!/usr/bin/env python3
"""
Start Togyzkumalak server from Jupyter Notebook
Run this in a Jupyter cell
"""

import subprocess
import os
import time
import requests
from IPython.display import display, IFrame, clear_output

# Change to project directory
os.chdir('/workspace/togyzkumalak/togyzkumalak-engine')

# Check if server is already running
try:
    response = requests.get('http://localhost:8000/api/health', timeout=2)
    if response.status_code == 200:
        display(HTML('<h3 style="color:green">✅ Server is already running!</h3>'))
        display(IFrame(src="http://localhost:8000", width="100%", height=800))
    else:
        raise Exception("Server not responding")
except:
    display(HTML('<h3 style="color:orange">🚀 Starting server...</h3>'))
    
    # Start server in background
    process = subprocess.Popen(
        ['python', 'run.py'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd='/workspace/togyzkumalak/togyzkumalak-engine'
    )
    
    # Wait for server to start
    for i in range(10):
        time.sleep(1)
        try:
            response = requests.get('http://localhost:8000/api/health', timeout=1)
            if response.status_code == 200:
                display(HTML('<h3 style="color:green">✅ Server started successfully!</h3>'))
                display(IFrame(src="http://localhost:8000", width="100%", height=800))
                break
        except:
            if i == 9:
                display(HTML('<h3 style="color:red">❌ Server failed to start. Check terminal for errors.</h3>'))
            continue

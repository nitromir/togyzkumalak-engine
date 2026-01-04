import os
from google import genai

api_key = "AIzaSyD7tqtEW1C-nMN-PC6OrjYotm8hsOWkdd8"
client = genai.Client(api_key=api_key)

try:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents="Hello"
    )
    print(f"SUCCESS: {response.text}")
except Exception as e:
    print(f"ERROR: {e}")

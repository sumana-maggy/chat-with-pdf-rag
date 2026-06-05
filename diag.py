import google.generativeai as genai
import os

# Try to get API key from environment or default
api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    # Try to extract from main.py
    try:
        with open("backend/main.py", "r") as f:
            for line in f:
                if "GEMINI_API_KEY =" in line:
                    api_key = line.split('"')[1]
                    break
    except:
        pass

if not api_key or api_key == "YOUR_GEMINI_API_KEY":
    print("API Key not found")
    exit(1)

genai.configure(api_key=api_key)

print("Listing models...")
try:
    models = genai.list_models()
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

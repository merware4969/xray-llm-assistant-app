# quick_gemini_test.py
import os, json, requests
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True), override=True)

base = os.environ.get("GEMINI_API_BASE","https://generativelanguage.googleapis.com").rstrip("/")
model = os.environ.get("LLM_MODEL","gemini-1.5-flash")
key = os.environ.get("GEMINI_API_KEY")
url = f"{base}/v1beta/models/{model}:generateContent?key={key}"
payload = {"contents":[{"role":"user","parts":[{"text":"ping"}]}]}
r = requests.post(url, json=payload, timeout=60)
print("status:", r.status_code)
print("body:", r.text[:1200])

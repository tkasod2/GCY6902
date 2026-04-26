import requests
import json
url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen2.5:3b",
    "prompt": "비트코인 자동매매에서 LLM의 역할을 3문장으로 설명해줘.",
    "stream": False
}

response = requests.post(url, json=payload)

print(response.json()["response"])
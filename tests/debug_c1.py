import httpx
import json

API_BASE = "https://hey-neha-shl-assessment-recommender.hf.space"

messages = []
turns = [
    "We need a solution for senior leadership.",
    "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
    "Selection — comparing candidates against a leadership benchmark.",
    "Perfect, that's what we need."
]

for i, turn in enumerate(turns, 1):
    messages.append({"role": "user", "content": turn})
    print(f"\n--- Turn {i} Request ---")
    print(f"User: {turn}")
    
    r = httpx.post(f"{API_BASE}/chat", json={"messages": messages})
    data = r.json()
    
    print(f"Reply: {data.get('reply')}")
    print(f"Recs: {data.get('recommendations')}")
    print(f"EOC: {data.get('end_of_conversation')}")
    
    messages.append({"role": "assistant", "content": data.get("reply", "")})

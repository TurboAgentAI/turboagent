"""
TurboAgent Cloud API Client — Example

Demonstrates connecting to a TurboAgent API server using the standard
OpenAI Python SDK. The server is a drop-in replacement for OpenAI's API.

Requirements:
    pip install openai

Server:
    turboagent serve --model Qwen/Qwen2.5-32B-Instruct --port 8000
"""

try:
    from openai import OpenAI
except ImportError:
    print("Install the OpenAI SDK: pip install openai")
    exit(1)

# Connect to TurboAgent server (same API as OpenAI)
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # Set your key if auth is enabled
)

# --- Single-turn completion ---

print("=== Single Turn ===")
response = client.chat.completions.create(
    model="turboagent",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is TurboQuant and why is it important for local AI?"},
    ],
    temperature=0.7,
    max_tokens=256,
)
print(response.choices[0].message.content)
print(f"\nUsage: {response.usage}")

# --- Multi-turn with session persistence ---

print("\n=== Multi-Turn (Session) ===")

# Use extra_headers for session ID (TurboAgent extension)
session_headers = {"X-Session-ID": "demo-session-1"}

# Turn 1
resp1 = client.chat.completions.create(
    model="turboagent",
    messages=[
        {"role": "user", "content": "Remember this: the project codename is PHOENIX-42."},
    ],
    extra_headers=session_headers,
)
print(f"Turn 1: {resp1.choices[0].message.content[:200]}")

# Turn 2 — server restores compressed KV cache from Turn 1
resp2 = client.chat.completions.create(
    model="turboagent",
    messages=[
        {"role": "user", "content": "Remember this: the project codename is PHOENIX-42."},
        {"role": "assistant", "content": resp1.choices[0].message.content},
        {"role": "user", "content": "What is the project codename?"},
    ],
    extra_headers=session_headers,
)
print(f"Turn 2: {resp2.choices[0].message.content[:200]}")

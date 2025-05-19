# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:43:13 2025

@author: Kuldeep
"""

# from groq_analysis import GroqAnalyzer
import pandas as pd

# Initialize with your Groq API key
API_KEY = "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"

# Load your DataFrame
your_df = pd.read_csv('2025-05-17T09-20_export.csv')  # or your DataFrame source


import requests

# --------------- CONFIGURATION ---------------
# API_KEY = "your_groq_api_key"  # 🔐 Replace with your real Groq API key
MODEL = "llama3-8b-8192"        # ✅ Try also: llama3-70b-8192, mixtral-8x7b-32768

API_URL = "https://api.groq.com/openai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# --------------- CHAT LOOP -------------------
conversation = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("💬 Groq Chat - Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Exiting chat. Goodbye!")
        break

    conversation.append({"role": "user", "content": user_input})

    payload = {
        "model": MODEL,
        "messages": conversation,
        "temperature": 0.7
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        reply = response.json()["choices"][0]["message"]["content"]
        print(f"Groq: {reply}\n")
        conversation.append({"role": "assistant", "content": reply})
    else:
        print(f"❌ Error {response.status_code}: {response.text}")


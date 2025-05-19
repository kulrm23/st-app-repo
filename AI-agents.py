# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:59:31 2025

@author: Kuldeep
"""

import pandas as pd
import os

# Use OpenAI from the langchain_openai package (new recommended import)
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# STEP 1: Load your CSV into a pandas DataFrame
your_df = pd.read_csv("2025-05-17T09-20_export.csv")

# STEP 2: Configure OpenAI client to use Groq
API_KEY = "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"
os.environ["OPENAI_API_KEY"] = API_KEY

OpenAI.api_base = "https://api.groq.com/openai/v1"

# STEP 3: Create a LangChain agent for interacting with the DataFrame
llm = OpenAI(model_name="llama3-8b-8192", temperature=0.0)
agent = create_pandas_dataframe_agent(llm, your_df, verbose=True)

# STEP 4: Chat loop
print("🧠 Ask questions about your CSV. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        print("👋 Done.")
        break
    try:
        response = agent.invoke(query)  # Use invoke() instead of deprecated run()
        print(f"\n🗨️ {response}\n")
    except Exception as e:
        print(f"❌ Error: {e}")

# Optional: Print langchain core version to check
import langchain_core
print(f"LangChain Core version: {langchain_core.__version__}")

# In[]

# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:59:31 2025

@author: Kuldeep
"""

import pandas as pd
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# STEP 1: Load your CSV into a pandas DataFrame
your_df = pd.read_csv("2025-05-17T09-20_export.csv")

# STEP 2: Provide your OpenAI API key directly
OPENAI_API_KEY = "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"


# STEP 3: Create the OpenAI LLM client with your API key
llm = OpenAI(model_name="gpt-4", temperature=0.0, openai_api_key=OPENAI_API_KEY)

# STEP 4: Create the LangChain agent for interacting with the DataFrame
agent = create_pandas_dataframe_agent(llm, your_df, verbose=True)

# STEP 5: Chat loop
print("🧠 Ask questions about your CSV. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ['exit', 'quit']:
        print("👋 Done.")
        break
    try:
        response = agent.invoke({"input": query})  # use invoke, as run is deprecated
        print(f"\n🗨️ {response['output']}\n")
    except Exception as e:
        print(f"❌ Error: {e}")


# In[]




import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

# Load numeric CSV
df = pd.read_csv("2025-05-17T09-20_export.csv")

# Optional: limit rows/columns to avoid hitting token limits
df_sample = df.head(20)

# Convert to string for LLM
csv_text = df_sample.to_csv(index=False)

# Create prompt
prompt = f"""
You are a data analyst. Analyze the following CSV data and summarize key statistics, trends, and anything interesting:

{csv_text}
"""

# Initialize Groq LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768"
)

# Ask the LLM
response = llm.invoke([HumanMessage(content=prompt)])
print("Analysis:\n", response.content)

# In[]

import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os

import pandas as pd

# Load CSV
df = pd.read_csv("2025-05-17T09-20_export.csv")

# Convert each row to a natural language summary
def row_to_text(row):
    return ", ".join(f"{col} is {val}" for col, val in row.items())

texts = df.apply(row_to_text, axis=1).tolist()

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)

# FAISS index
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# ID-to-text map
id_to_text = {i: texts[i] for i in range(len(texts))}

import requests
import os

def query_groq(prompt, model="llama3-70b-8192"):
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a data analyst assistant."},
            {"role": "user", "content": prompt}
        ]
    }
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
    return res.json()["choices"][0]["message"]["content"]




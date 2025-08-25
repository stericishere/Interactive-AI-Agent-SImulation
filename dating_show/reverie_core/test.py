"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: test.py
Description: Test script for OpenRouter API integration.
"""
import json
import random
import requests
import time 
import os

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-openrouter-api-key-here")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-chat-v3-0324:free"

def openrouter_request(messages, model=MODEL_NAME, temperature=0.7, max_tokens=None):
  """
  Make a request to OpenRouter API
  """
  headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
  }
  
  data = {
    "model": model,
    "messages": messages
  }
  
  if temperature is not None:
    data["temperature"] = temperature
  if max_tokens is not None:
    data["max_tokens"] = max_tokens
  
  try:
    response = requests.post(
      f"{OPENROUTER_BASE_URL}/chat/completions",
      headers=headers,
      json=data
    )
    response.raise_for_status()
    
    result = response.json()
    return result["choices"][0]["message"]["content"]
  
  except requests.exceptions.RequestException as e:
    print(f"OpenRouter API Error: {e}")
    return "OpenRouter API Error"
  except (KeyError, IndexError) as e:
    print(f"OpenRouter Response Error: {e}")
    return "OpenRouter Response Error"

def ChatGPT_request(prompt): 
  """
  Given a prompt, make a request to OpenRouter server and returns the response.
  ARGS:
    prompt: a str prompt
  RETURNS: 
    a str of llama-4-maverick's response. 
  """
  try: 
    messages = [{"role": "user", "content": prompt}]
    return openrouter_request(messages, MODEL_NAME)
  
  except: 
    print ("OpenRouter ERROR")
    return "OpenRouter ERROR"

prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

print (ChatGPT_request(prompt))













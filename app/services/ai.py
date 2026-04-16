from openai import OpenAI
import os
from dotenv import load_dotenv
from services.memory import build_system_prompt

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def get_ai_response(chat_history, memory):
    system_prompt = build_system_prompt(memory)

    messages = [{"role": "system", "content": system_prompt}]
    messages += chat_history[-10:]

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=messages,
    )

    return response.choices[0].message.content
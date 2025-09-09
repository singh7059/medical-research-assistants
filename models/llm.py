from groq import Groq
from config.config import GROQ_API_KEY

def get_llm_client():
    return Groq(api_key=GROQ_API_KEY)

def generate_llm_response(prompt, model="llama-3.1-8b-instant"):
    client = get_llm_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


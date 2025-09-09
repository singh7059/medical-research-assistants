import requests
from config.config import TAVILY_API_KEY

def tavily_search(query):
    url = "https://api.tavily.com/search"
    payload = {"api_key": TAVILY_API_KEY, "query": query}
    response = requests.post(url, json=payload)
    results = response.json()
    return [r["content"] for r in results.get("results", [])]

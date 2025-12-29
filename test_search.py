from duckduckgo_search import DDGS
import json

print("Testing DuckDuckGo Search...")
try:
    results = DDGS().text("current US president", max_results=3)
    print(f"Results found: {len(list(results))}")
    for r in results:
        print(r)
except Exception as e:
    print(f"Error: {e}")

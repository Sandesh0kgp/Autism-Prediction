"""
Chatbot module with context-aware responses using Groq API.
Integrates user profile, prediction history, and web search with smart relevance filtering.
"""

import os
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
from duckduckgo_search import DDGS
from googlesearch import search as google_search
import re

# Load environment variables
load_dotenv()


def init_chatbot() -> Optional[Groq]:
    """Initialize the chatbot with API key."""
    # Try getting key from Streamlit secrets (Cloud Deployment)
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except (ImportError, FileNotFoundError):
        api_key = None

    # Fallback to Local Environment Variable
    if not api_key:
        api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        return None
    
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
        return None


def build_context(user_profile: Dict, latest_prediction: Optional[Dict], 
                  history: List[Dict]) -> str:
    """Build context string from user data."""
    context_parts = []
    
    if user_profile:
        context_parts.append(f"User Profile:")
        context_parts.append(f"- Name: {user_profile['name']}")
        context_parts.append(f"- Age: {user_profile['age']}")
    
    if latest_prediction:
        pred_value = latest_prediction.get('prediction')
        if pred_value:
            context_parts.append(f"\nLatest Prediction:")
            context_parts.append(f"- Result: {pred_value}")
            
            if latest_prediction.get('input_data'):
                input_data = latest_prediction['input_data']
                context_parts.append(f"- Demographics & History:")
                context_parts.append(f"  • Gender: {input_data.get('gender', 'N/A')}")
                context_parts.append(f"  • Ethnicity: {input_data.get('ethnicity', 'N/A')}")
                context_parts.append(f"  • Country: {input_data.get('contry_of_res', 'N/A')}")
                context_parts.append(f"  • Jaundice at birth: {input_data.get('jaundice', 'N/A')}")
                context_parts.append(f"  • Family member with autism: {input_data.get('austim', 'N/A')}")
                context_parts.append(f"  • Used app before: {input_data.get('used_app_before', 'N/A')}")
                
                a_scores = [input_data.get(f'A{i}_Score', 0) for i in range(1, 11)]
                context_parts.append(f"- Screening Scores (A1-A10): {a_scores}")
    
    chat_history = [h for h in history if h.get('user_question')]
    if chat_history:
        context_parts.append(f"\nRecent Conversations:")
        for chat in chat_history[:3]:
            context_parts.append(f"- Q: {chat['user_question']}")
            if chat.get('bot_response'):
                response = chat['bot_response']
                if len(response) > 100:
                    response = response[:100] + "..."
                context_parts.append(f"  A: {response}")
    
    return "\n".join(context_parts)


def calculate_relevance(query: str, text: str) -> bool:
    """Check if result text contains relevant keywords."""
    text_lower = text.lower()
    query_lower = query.lower()
    
    # 1. Critical Filter: If query focuses on 'autism', result MUST mention it.
    if 'autism' in query_lower or 'asd' in query_lower:
        if 'autism' not in text_lower and 'asd' not in text_lower and 'spectrum' not in text_lower:
            return False

    # 2. Relaxed Filter
    return True


def search_web(query: str, max_results: int = 10) -> str:
    """Search the web with News backend support and Google Fallback."""
    clean_query = query.replace('"', '').replace("'", "")
    
    print(f"🔎 DEBUG: Attempting to search web for: '{clean_query}'")
    
    formatted_results = []
    
    # Check if this is a "News" query
    is_news = any(w in clean_query.lower() for w in ['latest', 'news', 'recent', 'today', 'update', 'current'])
    
    # --- STRATEGY 1: DuckDuckGo ---
    print("🔎 DEBUG: Using DuckDuckGo...")
    ddg_raw_results = []
    try:
        ddgs = DDGS()
        
        if is_news:
             # Try News First
             try:
                 print("🔎 DEBUG: Using DDG News backend...")
                 ddg_raw_results = list(ddgs.news(clean_query, max_results=max_results))
             except Exception:
                 print("🔎 DEBUG: DDG News failed, falling back to Text...")
                 ddg_raw_results = list(ddgs.text(clean_query, max_results=max_results))
        else:
             # Standard Text Search
             try:
                 ddg_raw_results = list(ddgs.text(clean_query, max_results=max_results, region='wt-wt'))
             except Exception:
                 print("🔎 DEBUG: DDG Default failed, trying Lite...")
                 ddg_raw_results = list(ddgs.text(clean_query, max_results=max_results, backend='lite'))
            
    except Exception as e:
        print(f"🔎 DEBUG: DuckDuckGo failed: {e}")

    # Start filtering DDG results
    valid_results = []
    if ddg_raw_results:
        print(f"🔎 DEBUG: Found {len(ddg_raw_results)} results via DuckDuckGo. Filtering...")
        for res in ddg_raw_results:
            title = res.get('title', '')
            body = res.get('body', '') if 'body' in res else res.get('date', '') 
            
            # Check relevance
            if calculate_relevance(clean_query, title) or calculate_relevance(clean_query, body):
                 valid_results.append(res)
            else:
                 print(f"   🗑️ Discarded irrelevant: {title[:30]}...")
    
    if valid_results:
        print(f"🔎 DEBUG: Kept {len(valid_results)} RELEVANT DDG results.")
        for i, result in enumerate(valid_results[:5], 1): 
            formatted_results.append(
                f"{i}. {result.get('title')}\n   {result.get('body')}\n   Source: {result.get('href') or result.get('url')}"
            )
        return "\n\n".join(formatted_results)

    # --- STRATEGY 2: Google Fallback ---
    print("🔎 DEBUG: DDG yielded no relevant results. Switching to Google Search Fallback...")
    try:
        g_results = list(google_search(clean_query, num_results=5, advanced=True, lang="en"))
        
        if g_results:
            print(f"🔎 DEBUG: Found {len(g_results)} results via Google")
            for i, result in enumerate(g_results, 1):
                formatted_results.append(
                    f"{i}. {result.title}\n   {result.description}\n   Source: {result.url}"
                )
            return "\n\n".join(formatted_results)
        else:
            print("🔎 DEBUG: Google search yielded 0 results.")
            
    except Exception as e:
        print(f"🔎 DEBUG: Google search failed: {e}")
            
    print("❌ DEBUG: All search methods failed or found no relevant info.")
    return "No relevant search results found."


def get_response(user_question: str, context: str, use_search: bool = True) -> Tuple[str, Optional[str]]:
    """Generate chatbot response using Groq API."""
    search_results = None
    try:
        client = init_chatbot()
        if not client:
            return (
                "⚠️ Chatbot not configured. ERROR: API Key missing or invalid.\n"
                "Please check chatbot.py settings.",
                None
            )
        
        # Trigger search for almost any question
        triggers = ['who', 'what', 'where', 'when', 'why', 'how', 'explain', 'tell', 'latest', 'current', 'news']
        needs_search = use_search and any(trigger in user_question.lower() for trigger in triggers)
        
        # System message
        system_message = (
            "You are a helpful assistant for an autism screening application. "
            "You have access to real-time web search results provided in the context. "
            "\n\nSEARCH RULES:"
            "\n1. If 'Web Search Results' contains valid information, YOU MUST USE IT to answer."
            "\n2. ALWAYS cite the sources from the results (e.g., 'According to...')."
            "\n3. If 'Web Search Results' says 'No relevant search results found', "
            "fallback to your internal training knowledge but state: 'My web search didn't find specific results, but I can tell you that...'."
            "\n\nGENERAL RULES:"
            "\n- Provide clear, empathetic responses."
            "\n- For medical topics, remind users to consult healthcare professionals."
        )
        
        user_message_parts = []
        
        if context:
            user_message_parts.append(f"User Context:\n{context}\n")
        
        if needs_search:
            print(f"🔎 DEBUG: Triggering search for: {user_question}")
            search_results_text = search_web(user_question)
            
            if "No relevant search results found" in search_results_text:
                 search_results = None
                 user_message_parts.append(f"Web Search Status: Search attempted but found no relevant information.\n")
            else:
                 search_results = search_results_text
                 user_message_parts.append(f"Web Search Results:\n{search_results}\n")
        else:
            print("🔎 DEBUG: Search skipped (no keyword trigger)")
        
        user_message_parts.append(f"User Question: {user_question}")
        
        user_message = "\n".join(user_message_parts)
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
        )
        
        return chat_completion.choices[0].message.content, search_results
        
    except Exception as e:
        error_msg = str(e)
        if "model_decommissioned" in error_msg:
             return "⚠️ The AI model is currently unavailable.", None
        return f"Error generating response: {error_msg}\n\nPlease check your internet connection.", None


def get_quick_response(user_question: str, user_name: str = None) -> str:
    """Get a quick response without full context (for testing)."""
    context = f"User name: {user_name}" if user_name else ""
    response, _ = get_response(user_question, context, use_search=True)
    return response

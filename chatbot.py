"""
Chatbot module with context-aware responses using Groq API.
Integrates user profile, prediction history, and web search with smart relevance filtering.
"""

import os
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
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


def search_web_with_perplexity(query: str) -> str:
    """
    Search the web using Perplexity AI API.
    Perplexity is an LLM with built-in web search that returns cited answers.
    """
    clean_query = query.replace('"', '').replace("'", "")
    
    # Add "autism" context if not present
    if 'autism' not in clean_query.lower() and 'asd' not in clean_query.lower():
        search_query = f"{clean_query} related to autism"
    else:
        search_query = clean_query

    print(f"🔎 DEBUG: Using Perplexity AI to search for: '{search_query}'")
    
    try:
        # Get Perplexity API key
        perplexity_key = os.getenv('PERPLEXITY_API_KEY')
        
        # Try Streamlit secrets if env var not found
        if not perplexity_key:
            try:
                import streamlit as st
                perplexity_key = st.secrets.get("PERPLEXITY_API_KEY")
            except (ImportError, FileNotFoundError, KeyError):
                pass
        
        if not perplexity_key:
            print("⚠️ DEBUG: Perplexity API key not found")
            return "Search unavailable: API key not configured."
        
        # Initialize Perplexity client (uses OpenAI SDK with custom base URL)
        client = OpenAI(
            api_key=perplexity_key,
            base_url="https://api.perplexity.ai"
        )
        
        # Call Perplexity with search-enabled model
        response = client.chat.completions.create(
            model="sonar",  # Updated model name for web search
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant. Provide concise, factual information with citations."
                },
                {
                    "role": "user",
                    "content": search_query
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        
        result = response.choices[0].message.content
        print(f"✅ DEBUG: Perplexity search successful, got {len(result)} chars")
        
        return f"Web Search Results (via Perplexity AI):\n\n{result}"
        
    except Exception as e:
        print(f"❌ DEBUG: Perplexity search failed: {str(e)}")
        return "Search failed. Please rely on internal knowledge."


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
        triggers = ['who', 'what', 'where', 'when', 'why', 'how', 'explain', 'tell', 'latest', 'current', 'news', 'research', 'treatment', 'study', 'development', 'treatments', 'finding', 'cause']
        needs_search = use_search and any(trigger in user_question.lower() for trigger in triggers)
        
        # System message
        system_message = (
            "You are a helpful assistant for an autism screening application. "
            "You have access to web-enhanced information via Perplexity AI provided in the context. "
            "\n\nSEARCH RULES:"
            "\n1. If 'Web Search Results' contains information from Perplexity AI, USE IT to enhance your answer."
            "\n2. Perplexity results already include citations, so integrate them naturally into your response."
            "\n3. If search results are unavailable or failed, use your internal knowledge and state: "
            "'My search didn't find current results, but based on my training...'."
            "\n\nGENERAL RULES:"
            "\n- Provide clear, empathetic responses."
            "\n- For medical topics, remind users to consult healthcare professionals."
        )
        
        user_message_parts = []
        
        if context:
            user_message_parts.append(f"User Context:\n{context}\n")
        
        if needs_search:
            print(f"🔎 DEBUG: Triggering Perplexity search for: {user_question}")
            search_results_text = search_web_with_perplexity(user_question)
            
            if "Search failed" in search_results_text or "Search unavailable" in search_results_text:
                 search_results = None
                 user_message_parts.append(f"Web Search Status: Search attempted but was unavailable.\n")
            else:
                 search_results = search_results_text
                 user_message_parts.append(f"{search_results}\n")
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

"""
Main Streamlit application for Autism Prediction with User Profiles and Chatbot.
Portfolio-ready ML application with clean UI and full data flow.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import traceback
import os

# Import custom modules
import database as db
import model_utils
import chatbot

# Enable debugging mode
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'


# Page configuration
st.set_page_config(
    page_title="Autism Screening App",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


def init_session_state():
    """Initialize session state variables."""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_name' not in st.session_state:
        st.session_state.user_name = None
    if 'user_age' not in st.session_state:
        st.session_state.user_age = None


def show_sidebar():
    """Display sidebar with navigation and user info."""
    with st.sidebar:
        st.title("🧠 Autism Screening")
        st.markdown("---")
        
        # Debug info
        if DEBUG_MODE:
            with st.expander("🐛 Debug Info", expanded=False):
                st.caption("Model files exist:")
                st.write(f"- best_model.pkl: {os.path.exists('best_model.pkl')}")
                st.write(f"- encoders.pkl: {os.path.exists('encoders.pkl')}")
                st.caption(f"Database: {os.path.exists('autism.db')}")
                st.caption(f"Session State: {dict(st.session_state)}")
        
        # Show user info if logged in
        if st.session_state.user_id:
            st.success(f"👤 Logged in as: **{st.session_state.user_name}**")
            st.caption(f"Age: {st.session_state.user_age}")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.user_id = None
                st.session_state.user_name = None
                st.session_state.user_age = None
                st.rerun()
        else:
            st.info("👋 Please create a profile to get started")
        
        st.markdown("---")
        
        # Navigation
        st.subheader("Navigation")
        
        pages = {
            "Create Profile": "👤",
            "Predict": "🔮",
            "Chatbot": "💬",
            "History": "📊"
        }
        
        selected_page = st.radio(
            "Go to:",
            list(pages.keys()),
            format_func=lambda x: f"{pages[x]} {x}",
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.caption("Built with Streamlit • Portfolio Project")
        
        return selected_page


def page_create_profile():
    """User profile creation page."""
    st.title("👤 Create User Profile")
    st.markdown("Create a profile to start using the autism screening tool.")
    
    with st.form("profile_form"):
        st.subheader("Your Information")
        
        name = st.text_input("Name", placeholder="Enter your name")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        
        submitted = st.form_submit_button("Create Profile", use_container_width=True)
        
        if submitted:
            if not name:
                st.error("❌ Please enter your name")
            else:
                # Create user
                user_id = db.create_user(name, age)
                
                # Update session state
                st.session_state.user_id = user_id
                st.session_state.user_name = name
                st.session_state.user_age = age
                
                st.success(f"✅ Profile created successfully!")
                st.info(f"🆔 Your User ID: `{user_id}`")
                st.balloons()
                
                # Prompt to navigate
                st.markdown("👉 Use the sidebar to navigate to **Predict** or **Chatbot**")


def page_predict():
    """Model prediction page."""
    st.title("🔮 Autism Screening Prediction")
    
    # Check if user is logged in
    if not st.session_state.user_id:
        st.warning("⚠️ Please create a profile first")
        return
    
    st.markdown(f"Making prediction for: **{st.session_state.user_name}**")
    
    # Get feature descriptions
    features = model_utils.get_feature_names()
    
    with st.form("prediction_form"):
        st.subheader("📋 Screening Questions")
        st.caption("Answer the following questions (0 = No, 1 = Yes)")
        
        # A-Scores in a grid
        cols = st.columns(2)
        a_scores = {}
        
        for i in range(1, 11):
            col_idx = (i - 1) % 2
            with cols[col_idx]:
                question = features[f'A{i}_Score'].replace(f'A{i}: ', '')
                a_scores[f'A{i}_Score'] = st.selectbox(
                    f"**Q{i}**: {question}",
                    options=[0, 1],
                    key=f'a{i}'
                )
        
        st.markdown("---")
        st.subheader("👤 Demographic Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, 
                                 value=st.session_state.user_age)
            
            gender = st.selectbox("Gender", options=['m', 'f'])
            
            jaundice = st.selectbox("Born with jaundice?", options=['no', 'yes'])
            
            austim = st.selectbox("Family member with autism?", options=['no', 'yes'])
        
        with col2:
            ethnicity = st.selectbox(
                "Ethnicity",
                options=['White-European', 'Asian', 'Middle Eastern', 'Black', 
                        'Hispanic', 'Latino', 'South Asian', 'Pasifika', 
                        'Turkish', 'others', 'Others', '?']
            )
            
            country = st.text_input("Country of residence", value="United States")
            
            used_app = st.selectbox("Used screening app before?", options=['no', 'yes'])
        
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = {
                **a_scores,
                'age': age,
                'gender': gender,
                'ethnicity': ethnicity,
                'jaundice': jaundice,
                'austim': austim,
                'contry_of_res': country,
                'used_app_before': used_app
            }
            
            # Debug: Show input data
            if DEBUG_MODE:
                with st.expander("🔍 Input Data (Debug)"):
                    st.json(input_data)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                try:
                    result = model_utils.predict(input_data)
                    
                    if DEBUG_MODE:
                        with st.expander("🔍 Prediction Result (Debug)"):
                            st.json(result)
                            
                except Exception as e:
                    st.error(f"❌ Prediction Error: {str(e)}")
                    if DEBUG_MODE:
                        st.code(traceback.format_exc())
                    result = {'error': str(e), 'result': f'Error: {str(e)}'}
                    return
            
            # Display result
            if 'error' in result:
                st.error(f"❌ {result['result']}")
            else:
                # Save to database
                db.save_prediction(
                    st.session_state.user_id,
                    input_data,
                    result['result']
                )
                
                # Display result
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                if result['prediction'] == 1:
                    st.error(f"⚠️ {result['result']}")
                else:
                    st.success(f"✅ {result['result']}")
                
                # Show probabilities if available
                if result.get('probabilities'):
                    st.markdown("**Confidence Scores:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("No ASD", f"{result['probabilities']['no_autism']:.1%}")
                    with col2:
                        st.metric("ASD Traits", f"{result['probabilities']['autism']:.1%}")
                
                st.info("💡 **Note**: This is a screening tool, not a diagnosis. Please consult a healthcare professional for proper evaluation.")
                
                # Show input summary
                with st.expander("📝 View Input Summary"):
                    st.json(input_data)


def page_chatbot():
    """Chatbot page."""
    st.title("💬 AI Chatbot Assistant")
    
    # Check if user is logged in
    if not st.session_state.user_id:
        st.warning("⚠️ Please create a profile first")
        return
    
    st.caption(f"Chatting as: **{st.session_state.user_name}**")
    
    # Get user data for context
    user_profile = db.get_user(st.session_state.user_id)
    latest_prediction = db.get_latest_prediction(st.session_state.user_id)
    history = db.get_user_history(st.session_state.user_id)
    
    # Build context
    context = chatbot.build_context(user_profile, latest_prediction, history)
    
    # Show context only in debug mode or if requested, to keep UI clean
    with st.expander("🔍 What the AI knows about you (Context)", expanded=False):
        st.text(context)
        
    st.divider()

    # Display chat history (REVERSED to show oldest first -> newest last)
    # Filter only chat items
    chat_history = [h for h in history if h.get('user_question')]
    
    # Sort by timestamp ascending (oldest first) so it reads like a conversation
    chat_history.sort(key=lambda x: x['timestamp'])
    
    # Display all messages
    for chat in chat_history:
        with st.chat_message("user"):
            st.write(chat['user_question'])
        with st.chat_message("assistant"):
            st.write(chat['bot_response'])
    
    # Chat input area (at the bottom)
    if prompt := st.chat_input("Ask a question about your results or autism..."):
        # User message
        with st.chat_message("user"):
            st.write(prompt)
            
        # Bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get response
                    # Default use_search to True for better answers
                    # Returns (response_text, search_sources_text)
                    response, sources = chatbot.get_response(prompt, context, use_search=True)
                    st.write(response)
                    
                    # Show sources if any (Confirmation that DuckDuckGo works!)
                    if sources:
                        with st.expander("📚 View Search Sources (DuckDuckGo Results)"):
                            st.info("The chatbot used these search results:")
                            st.text(sources)
                    
                    # Save interaction to database
                    db.save_chat(st.session_state.user_id, prompt, response)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Rerun to update the history view
        st.rerun()


def page_history():
    """User history page."""
    st.title("📊 Your Activity History")
    
    # Check if user is logged in
    if not st.session_state.user_id:
        st.warning("⚠️ Please create a profile first")
        return
    
    st.markdown(f"Viewing history for: **{st.session_state.user_name}**")
    
    # Get history
    history = db.get_user_history(st.session_state.user_id)
    
    if not history:
        st.info("📭 No activity yet. Make a prediction or chat with the bot!")
        return
    
    # Separate predictions and chats
    predictions = [h for h in history if h.get('prediction')]
    chats = [h for h in history if h.get('user_question')]
    
    # Display predictions
    st.subheader(f"🔮 Predictions ({len(predictions)})")
    
    if predictions:
        for i, pred in enumerate(predictions, 1):
            with st.expander(f"Prediction #{i} - {pred['timestamp'][:19]}"):
                st.markdown(f"**Result:** {pred['prediction']}")
                
                if pred.get('input_data'):
                    st.markdown("**Input Data:**")
                    st.json(pred['input_data'])
    else:
        st.caption("No predictions yet")
    
    st.markdown("---")
    
    # Display chats
    st.subheader(f"💬 Chat History ({len(chats)})")
    
    if chats:
        for i, chat in enumerate(chats, 1):
            with st.expander(f"Chat #{i} - {chat['timestamp'][:19]}"):
                st.markdown(f"**Q:** {chat['user_question']}")
                st.markdown(f"**A:** {chat['bot_response']}")
    else:
        st.caption("No chat history yet")


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Show sidebar and get selected page
    selected_page = show_sidebar()
    
    # Route to selected page
    if selected_page == "Create Profile":
        page_create_profile()
    elif selected_page == "Predict":
        page_predict()
    elif selected_page == "Chatbot":
        page_chatbot()
    elif selected_page == "History":
        page_history()


if __name__ == "__main__":
    main()

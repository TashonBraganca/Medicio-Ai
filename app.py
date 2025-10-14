#!/usr/bin/env python3
"""
MediLens Local - Medical AI Assistant
Complete medical AI system with chat, document analysis, and image triage
"""

import streamlit as st
import requests
import json
import os
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from config import config

# Import services
try:
    from services.chat_service import MedicalChatService
    from services.safety_guard import SafetyGuard
    from services.document_service import DocumentProcessor
    from services.vision_service import VisionAnalyzer
    SERVICES_AVAILABLE = True
except ImportError as e:
    SERVICES_AVAILABLE = False
    st.error(f"Services not available: {e}")

# Initialize persistent Ollama manager SILENTLY (only once) - NO SPLASH SCREEN
if 'ollama_manager' not in st.session_state:
    from services.ollama_manager import ollama_manager
    st.session_state.ollama_manager = ollama_manager

    # Start persistent server SILENTLY in background - NO MESSAGES
    if not ollama_manager.startup_complete:
        # Start silently without any UI messages
        success, message = ollama_manager.start_persistent_server()
        # Store status but don't display anything - direct to main page

# Initialize services
if SERVICES_AVAILABLE:
    chat_service = MedicalChatService()
    safety_guard = SafetyGuard()
    document_processor = DocumentProcessor()
    vision_analyzer = VisionAnalyzer()

# Page configuration
st.set_page_config(
    page_title="MediLens Local",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "chat"
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "User"
    if 'processing_message' not in st.session_state:
        st.session_state.processing_message = False
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = config.DEFAULT_LLM_MODEL
    if 'model_search_query' not in st.session_state:
        st.session_state.model_search_query = ""

def check_ollama_status():
    """Check if Ollama service is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def render_sidebar():
    """Render informational sidebar."""
    with st.sidebar:
        st.markdown("# 🏥 MediLens Local")
        st.markdown("*Your Private Medical AI Assistant*")

        # Navigation
        st.markdown("---")

        if st.button("💬 Medical Chat", key="nav_chat", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()

        if st.button("📄 Report Analysis", key="nav_docs", use_container_width=True):
            st.session_state.current_page = "documents"
            st.rerun()

        if st.button("🖼️ Image Analysis", key="nav_images", use_container_width=True):
            st.session_state.current_page = "images"
            st.rerun()

        # System Status
        st.markdown("---")
        ollama_running, models_data = check_ollama_status()
        if ollama_running:
            st.success("✅ System Online")

            # Show active model
            if SERVICES_AVAILABLE and chat_service:
                active_model = chat_service.model
                st.info(f"🤖 Active: {active_model}")

            if models_data and 'models' in models_data:
                model_count = len(models_data['models'])
                st.caption(f"📦 {model_count} models available")

                # Show model optimization status - prioritize medical-specific models
                models = [model['name'] for model in models_data['models']]
                if any('meditron:7b' in model for model in models):
                    st.success("👑 MEDICAL EXPERT Mode (meditron:7b - Trained on medical literature)")
                elif any('llama3.1:8b' in model for model in models):
                    st.success("⭐ Premium Mode Available (llama3.1:8b)")
                elif any('mistral:7b' in model for model in models):
                    st.success("⭐ High-Quality Mode Available (mistral:7b)")
                elif any('gemma2:2b' in model for model in models):
                    st.success("⚡ Fast Mode Active (gemma2:2b)")
                elif any('qwen2:1.5b' in model for model in models):
                    st.success("🚀 Ultra-Fast Mode (qwen2:1.5b)")
                else:
                    st.warning("⚠️ Recommended models not detected")
        else:
            st.error("❌ System Offline")

        # Medical Information
        st.markdown("---")
        with st.expander("🩺 Medical Guidelines", expanded=False):
            st.markdown("""
            **Important:**
            - This is AI assistance, not medical diagnosis
            - Always consult healthcare professionals
            - For emergencies, call emergency services
            - AI responses are preliminary guidance only

            **Capabilities:**
            - Symptom assessment and guidance
            - Medical information and explanations
            - Treatment suggestions and care tips
            - Lab result interpretation assistance
            """)

        with st.expander("⚙️ Settings", expanded=False):
            # Enhanced Model Selection with Search
            st.markdown("#### 🤖 AI Model Selection")

            # Search input
            search_query = st.text_input(
                "🔍 Search Models",
                value=st.session_state.model_search_query,
                placeholder="Type to filter models...",
                key="model_search_input",
                help="Search for models by name"
            )
            st.session_state.model_search_query = search_query

            # Get all models with details
            if models_data and 'models' in models_data:
                all_models = models_data['models']

                # Filter models based on search
                if search_query.strip():
                    filtered_models = [
                        m for m in all_models
                        if search_query.lower() in m['name'].lower()
                    ]
                else:
                    filtered_models = all_models

                if filtered_models:
                    # Display filtered models count
                    st.caption(f"Found {len(filtered_models)} model(s)")

                    # Radio buttons for selection
                    model_names = [m['name'] for m in filtered_models]

                    # Get current index
                    current_index = 0
                    if st.session_state.selected_model in model_names:
                        current_index = model_names.index(st.session_state.selected_model)

                    selected_model = st.radio(
                        "Select Model:",
                        options=model_names,
                        index=current_index,
                        key="model_radio"
                    )

                    # Show model details
                    selected_model_data = next((m for m in filtered_models if m['name'] == selected_model), None)
                    if selected_model_data:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Convert size to GB/MB
                            size_bytes = selected_model_data.get('size', 0)
                            size_gb = size_bytes / (1024**3)
                            if size_gb >= 1:
                                st.caption(f"📦 {size_gb:.2f} GB")
                            else:
                                size_mb = size_bytes / (1024**2)
                                st.caption(f"📦 {size_mb:.0f} MB")

                        with col2:
                            # Show modified date (simplified)
                            st.caption(f"🕐 Available")

                    # Apply button
                    current_active = chat_service.model if SERVICES_AVAILABLE else config.DEFAULT_LLM_MODEL

                    if selected_model != current_active:
                        if st.button("✅ Apply Model", use_container_width=True, type="primary", key="apply_model_btn"):
                            if SERVICES_AVAILABLE:
                                # Update chat service model
                                chat_service.model = selected_model
                                st.session_state.selected_model = selected_model
                                st.success(f"✅ Switched to {selected_model}")
                                time.sleep(0.5)  # Brief delay for user to see message
                                st.rerun()
                            else:
                                st.error("Services not available")
                    else:
                        st.info(f"✓ Active: {selected_model}")

                else:
                    st.warning(f"No models matching '{search_query}'")
            else:
                st.error("No models available")

            st.markdown("---")

            # Temperature setting
            temperature = st.slider(
                "Response Style",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Lower = More focused, Higher = More creative"
            )

            # Clear chat
            if st.button("🗑️ Clear Conversation", use_container_width=True):
                st.session_state.chat_history = []
                st.success("Conversation cleared!")
                st.rerun()

        # Privacy Notice
        st.markdown("---")
        st.markdown("### 🔒 Privacy First")
        st.caption("100% local processing - your data never leaves your device")

# NUCLEAR CSS OVERHAUL - ULTIMATE BACKGROUND AND STYLING FIX
def load_css():
    """Load nuclear-level CSS for medical interface."""
    st.markdown("""
    <style>
    /* Hide Streamlit defaults - BUT KEEP HEADER FOR SIDEBAR CONTROLS */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    .stToolbar {display: none;}

    /* Keep header visible for sidebar collapse/resize - just hide the unwanted parts */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* Hide header hamburger menu but keep sidebar controls */
    header [data-testid="stHeader"] > div:first-child {
        display: none;
    }

    /* NUCLEAR BACKGROUND SOLUTION - FORCE GRADIENT ON HTML/BODY FIRST */
    html {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        min-height: 100vh !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    body {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        min-height: 100vh !important;
        height: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        color: #f8fafc !important;
    }

    /* SELECTIVE TRANSPARENT OVERRIDE - PRESERVE SIDEBAR AND UI CONTRAST */
    .main, .main > div,
    [data-testid="stMainBlockContainer"],
    [data-testid="stVerticalBlock"],
    [data-testid="stAppViewContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
    }

    /* STREAMLIT ROOT CONTAINER - FORCE GRADIENT */
    #root, [data-testid="stApp"], .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        min-height: 100vh !important;
        height: auto !important;
        color: #f8fafc !important;
    }

    /* TARGETED MAIN CONTENT TRANSPARENCY - PRESERVE UI ELEMENTS */
    .main .block-container,
    [data-testid="stMain"] .block-container,
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
        background-color: transparent !important;
        background-image: none !important;
    }

    /* BOTTOM AREA NUCLEAR FIX */
    [data-testid="stChatInputContainer"], [data-testid="stChatInputContainer"] > div,
    [data-testid="stChatInputContainer"] > div > div, [data-testid="stChatInputContainer"] > div > div > div,
    .stChatInput, .stChatInput > div, .stChatInput > div > div,
    .stForm, [data-testid="stForm"], .stForm > div, [data-testid="stForm"] > div,
    .stBottom, [data-testid="stBottom"], .stBottom > div, [data-testid="stBottom"] > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }

    /* ENSURE GRADIENT REACHES BOTTOM - ULTIMATE FIX */
    [data-testid="stApp"]::after {
        content: "";
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 200px;
        background: linear-gradient(to top, #0f172a 50%, transparent 100%);
        z-index: -1;
        pointer-events: none;
    }

    /* Chat container with auto-scroll */
    #chat-container {
        max-height: 60vh;
        overflow-y: auto;
        scroll-behavior: smooth;
        padding: 1rem;
        margin-bottom: 1rem;
        background: transparent !important;
    }

    /* PROPER CHAT INPUT STYLING WITH CONTRAST */
    .stChatInput > div, [data-testid="stChatInput"] > div {
        background: rgba(30, 41, 59, 0.9) !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 24px !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.3) !important;
    }

    /* Chat input text */
    .stChatInput input {
        color: #f1f5f9 !important;
        background: transparent !important;
    }

    .stChatInput input::placeholder {
        color: #94a3b8 !important;
    }

    /* Medical floating background icons */
    @keyframes float1 {
        0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0.03; }
        25% { transform: translateY(-20px) translateX(10px) rotate(90deg); opacity: 0.05; }
        50% { transform: translateY(0) translateX(-10px) rotate(180deg); opacity: 0.03; }
        75% { transform: translateY(20px) translateX(5px) rotate(270deg); opacity: 0.05; }
    }

    @keyframes float2 {
        0%, 100% { transform: translateY(0) translateX(0) rotate(0deg); opacity: 0.03; }
        33% { transform: translateY(30px) translateX(-20px) rotate(120deg); opacity: 0.05; }
        66% { transform: translateY(-15px) translateX(15px) rotate(240deg); opacity: 0.03; }
    }

    .medical-bg {
        position: fixed;
        width: 100%;
        height: 100%;
        top: 0;
        left: 0;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .medical-icon {
        position: absolute;
        font-size: 80px;
        color: #60a5fa;
        opacity: 0.03;
    }

    .icon1 { top: 10%; left: 10%; animation: float1 20s infinite ease-in-out; }
    .icon2 { top: 20%; left: 80%; animation: float2 25s infinite ease-in-out; }
    .icon3 { top: 60%; left: 20%; animation: float1 30s infinite ease-in-out; }
    .icon4 { top: 70%; left: 70%; animation: float2 22s infinite ease-in-out; }
    .icon5 { top: 40%; left: 50%; animation: float1 28s infinite ease-in-out; }
    .icon6 { top: 85%; left: 40%; animation: float2 24s infinite ease-in-out; }

    /* Welcome container */
    .welcome-container {
        text-align: center;
        padding: 40px 20px;
        max-width: 700px;
        margin: 0 auto;
    }

    .welcome-title {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 16px;
        letter-spacing: -0.5px;
    }

    .welcome-subtitle {
        font-size: 18px;
        color: #94a3b8;
        margin-bottom: 30px;
        font-weight: 400;
    }

    /* Chat messages */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin: 16px 0;
        padding: 0 20px;
    }

    .assistant-message {
        display: flex;
        justify-content: flex-start;
        margin: 16px 0;
        padding: 0 20px;
    }

    .user-bubble {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 12px 16px;
        border-radius: 16px 16px 4px 16px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }

    .assistant-bubble {
        background: rgba(30, 41, 59, 0.9);
        color: #f1f5f9;
        padding: 12px 16px;
        border-radius: 16px 16px 16px 4px;
        max-width: 70%;
        word-wrap: break-word;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(100, 116, 139, 0.3);
    }

    /* Page content styling */
    .main-content {
        position: relative;
        z-index: 1;
        padding: 20px;
    }

    /* File uploader styling */
    .stFileUploader {
        background: rgba(30, 41, 59, 0.5);
        border: 2px dashed #60a5fa;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    /* PROPER MAIN CONTENT BUTTON STYLING */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3) !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(59, 130, 246, 0.4) !important;
    }

    /* PROPER SIDEBAR STYLING - DISTINCT FROM MAIN CONTENT */
    [data-testid="stSidebar"], .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%) !important;
        border-right: 1px solid #475569 !important;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.3) !important;
    }

    /* Sidebar content styling */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stSlider,
    [data-testid="stSidebar"] .stExpander {
        background: transparent;
    }

    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #475569 0%, #64748b 100%) !important;
        color: #f1f5f9 !important;
        border: 1px solid #64748b !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: linear-gradient(135deg, #64748b 0%, #475569 100%) !important;
        border-color: #94a3b8 !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar text */
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #f1f5f9 !important;
    }

    /* Sidebar success/info messages */
    [data-testid="stSidebar"] .stSuccess {
        background: rgba(34, 197, 94, 0.2) !important;
        border: 1px solid #22c55e !important;
        color: #bbf7d0 !important;
    }

    [data-testid="stSidebar"] .stInfo {
        background: rgba(59, 130, 246, 0.2) !important;
        border: 1px solid #3b82f6 !important;
        color: #dbeafe !important;
    }

    [data-testid="stSidebar"] .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        border: 1px solid #ef4444 !important;
        color: #fecaca !important;
    }

    /* Streamlit's built-in sidebar collapse works perfectly - no custom toggle needed */

    /* PROPER ALERT STYLING WITH CONTRAST */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    /* Success alerts */
    [data-testid="stAlert"][data-baseweb="notification"] div[data-testid="stNotificationContentSuccess"] {
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid #22c55e !important;
        color: #bbf7d0 !important;
    }

    /* Info alerts */
    [data-testid="stAlert"][data-baseweb="notification"] div[data-testid="stNotificationContentInfo"] {
        background: rgba(59, 130, 246, 0.15) !important;
        border: 1px solid #3b82f6 !important;
        color: #dbeafe !important;
    }

    /* Error alerts */
    [data-testid="stAlert"][data-baseweb="notification"] div[data-testid="stNotificationContentError"] {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 1px solid #ef4444 !important;
        color: #fecaca !important;
    }
    </style>
    """, unsafe_allow_html=True)

def render_medical_background():
    """Render floating medical icons in background."""
    st.markdown("""
    <div class="medical-bg">
        <div class="medical-icon icon1">🩺</div>
        <div class="medical-icon icon2">💊</div>
        <div class="medical-icon icon3">🏥</div>
        <div class="medical-icon icon4">❤️</div>
        <div class="medical-icon icon5">🔬</div>
        <div class="medical-icon icon6">🩹</div>
    </div>
    """, unsafe_allow_html=True)

def render_welcome_screen():
    """Render welcome screen for chat."""
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">How are you feeling today?</div>
        <div class="welcome-subtitle">I'm here to help with your medical questions</div>
    </div>
    """, unsafe_allow_html=True)

def render_chat_message(message, is_user=True):
    """Render a single chat message."""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            <div class="assistant-bubble">{message}</div>
        </div>
        """, unsafe_allow_html=True)

def process_user_message(user_message):
    """Process user message and generate AI response."""
    if not SERVICES_AVAILABLE:
        error_msg = "❌ Medical services are not available. Please check the services installation."
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        return

    # Test connection
    connected, status_msg = chat_service.test_connection()
    if not connected:
        error_msg = f"❌ Cannot connect to AI service: {status_msg}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
        return

    try:
        # SIMPLE NON-STREAMING APPROACH - GUARANTEED TO WORK
        with st.spinner("🤔 Thinking..."):
            # Get complete response from Ollama
            response_data = chat_service.get_medical_response(user_message, st.session_state.chat_history[:-1])

            if response_data["success"]:
                ai_response = response_data["response"]
                if isinstance(ai_response, dict):
                    # Format structured response
                    formatted_response = format_medical_response(ai_response)
                else:
                    formatted_response = str(ai_response)

                st.session_state.chat_history.append({"role": "assistant", "content": formatted_response})
            else:
                error_response = f"❌ Error: {response_data.get('error', 'Unknown error occurred')}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_response})

    except Exception as e:
        error_response = f"❌ Error processing query: {str(e)}"
        st.session_state.chat_history.append({"role": "assistant", "content": error_response})

def format_medical_response(response_dict):
    """Format structured medical response for display - Enhanced prompt handles formatting."""
    if not isinstance(response_dict, dict):
        return str(response_dict)

    # Get the full response from summary (already formatted by enhanced prompt)
    summary = response_dict.get("summary", "")

    if summary:
        # The new prompt includes proper formatting with emojis, sections, and structure
        # Just ensure clean line breaks and return
        formatted_response = summary

        # Clean up excessive line breaks
        formatted_response = formatted_response.replace("\n\n\n\n", "\n\n")
        formatted_response = formatted_response.replace("\n\n\n", "\n\n")
        formatted_response = formatted_response.strip()

        return formatted_response
    else:
        # Fallback if no summary available
        return "Unable to generate medical response. Please try again."

def render_chat_page():
    """Render the main chat page with auto-scroll functionality."""
    # Chat container with unique ID for auto-scroll
    st.markdown('<div class="main-content" id="chat-container">', unsafe_allow_html=True)

    # Welcome message or chat history
    if not st.session_state.chat_history:
        render_welcome_screen()
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                render_chat_message(message["content"], is_user=True)
            else:
                render_chat_message(message["content"], is_user=False)

    st.markdown('</div>', unsafe_allow_html=True)

    # NO AUTO-SCROLL JAVASCRIPT - Streamlit handles this natively

    # Chat input
    if prompt := st.chat_input("Ask me anything medical..."):
        # Add user message immediately to show it
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.session_state.processing_message = True

        # Rerun to display user message first
        st.rerun()

    # Process any pending user message
    if st.session_state.processing_message and st.session_state.chat_history:
        last_message = st.session_state.chat_history[-1]
        if last_message["role"] == "user":
            # Process message and generate response
            process_user_message(last_message["content"])
            st.session_state.processing_message = False
            # CRITICAL FIX: Rerun to display the AI response!
            st.rerun()

def render_documents_page():
    """Render document analysis page."""
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("📄 Medical Report Analysis")
    st.info("Upload medical documents, lab reports, or test results for AI-powered analysis and interpretation.")

    uploaded_file = st.file_uploader(
        "Choose a medical document",
        type=['pdf', 'jpg', 'jpeg', 'png'],
        help="Supports PDF documents and images"
    )

    if uploaded_file and SERVICES_AVAILABLE:
        with st.spinner("🔄 Processing document..."):
            file_bytes = uploaded_file.read()
            file_type = uploaded_file.type
            filename = uploaded_file.name

            # Process document
            processing_result = document_processor.process_document(file_bytes, file_type, filename)

        if processing_result["success"]:
            st.success("✅ Document processed successfully")

            # Show extracted text
            with st.expander("📄 Extracted Text", expanded=True):
                st.text_area("Content:", processing_result["extracted_text"], height=200, disabled=True)

            # AI Analysis
            st.subheader("🧠 AI Medical Analysis")
            with st.spinner("Analyzing document..."):
                analysis_result = document_processor.analyze_medical_document(processing_result["extracted_text"])

            if analysis_result["success"]:
                st.markdown(analysis_result["analysis"])
            else:
                st.error(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")

        else:
            st.error(f"Document processing failed: {processing_result.get('error', 'Unknown error')}")

    st.markdown('</div>', unsafe_allow_html=True)

def render_images_page():
    """Render image analysis page."""
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    st.title("🖼️ Medical Image Analysis")
    st.info("Upload medical images for AI-powered analysis and preliminary assessment.")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_image = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

    with col2:
        camera_image = st.camera_input("Take Photo")

    image_to_analyze = uploaded_image or camera_image

    if image_to_analyze and SERVICES_AVAILABLE:
        # Test vision model
        connected, status_msg = vision_analyzer.test_vision_model_connection()
        if not connected:
            st.error(f"❌ Vision model not available: {status_msg}")
            return

        # Display image
        st.image(image_to_analyze, caption="Medical Image", use_column_width=True)

        # Optional query
        user_query = st.text_input("Specific question about the image (optional):")

        # Analyze image
        if st.button("🔍 Analyze Image"):
            with st.spinner("🔍 Analyzing image..."):
                image_bytes = image_to_analyze.read()
                analysis_result = vision_analyzer.analyze_medical_image(
                    image_bytes,
                    user_query if user_query.strip() else None
                )

            if analysis_result["success"]:
                st.success("✅ Analysis completed")

                # Show analysis
                raw_response = analysis_result.get("raw_response", "")
                if raw_response:
                    st.markdown("### 🔍 Analysis Results")
                    st.write(raw_response)

                # Check for emergency indicators
                emergency_keywords = vision_analyzer.detect_emergency_indicators(raw_response)
                if emergency_keywords:
                    st.error(f"🚨 **URGENT INDICATORS DETECTED:** {', '.join(emergency_keywords)}")
                    st.error("**Seek immediate medical attention.**")

            else:
                st.error(f"Analysis failed: {analysis_result.get('error', 'Unknown error')}")

    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()

    # Load CSS and render background
    load_css()
    render_medical_background()

    # Render sidebar (using Streamlit's built-in collapse - it works perfectly!)
    render_sidebar()

    # Page routing
    if st.session_state.current_page == "chat":
        render_chat_page()
    elif st.session_state.current_page == "documents":
        render_documents_page()
    elif st.session_state.current_page == "images":
        render_images_page()
    else:
        render_chat_page()

if __name__ == "__main__":
    main()
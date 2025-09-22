#!/usr/bin/env python3
"""
MediLens Local - Chat Service
Core Ollama integration and medical chat functionality.
"""

import requests
import json
import time
import re
from typing import Dict, List, Any, Optional, Tuple, Generator
from datetime import datetime
import logging

from config import config
from services.safety_guard import SafetyGuard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalChatService:
    """Handles medical chat interactions with Ollama."""
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.timeout = config.OLLAMA_TIMEOUT
        self.retry_attempts = config.OLLAMA_RETRY_ATTEMPTS
        self.model = config.DEFAULT_LLM_MODEL  # ULTRA-FAST GEMMA2:2B
        self.safety_guard = SafetyGuard()
        
    def test_connection(self) -> Tuple[bool, str]:
        """Test connection to Ollama service and check for gemma2:2b model."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                installed_models = [model['name'] for model in data.get('models', [])]

                # Check if gemma2:2b is available (primary ultra-fast model)
                if any('gemma2:2b' in model for model in installed_models):
                    return True, f"Connected to Ollama with gemma2:2b ultra-fast model"
                # Check fallback model
                elif any('qwen2:1.5b' in model for model in installed_models):
                    return True, f"Connected to Ollama with qwen2:1.5b fallback model"
                # Legacy phi3:mini support
                elif any('phi3:mini' in model for model in installed_models):
                    return True, f"Connected to Ollama with phi3:mini model (legacy)"
                else:
                    return False, f"Ultra-fast models not found. Available models: {installed_models}"
            else:
                return False, f"Ollama service returned status code: {response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    
    def get_medical_response(self, user_message: str, chat_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Get structured medical response from Ollama."""
        try:
            # Check if query is medical
            if not self._is_medical_query(user_message):
                return {
                    "success": False,
                    "response": "I can help with medical and health-related questions. Please try rephrasing your question with medical context, symptoms, conditions, or health concerns.",
                    "error_type": "non_medical_query"
                }
            
            # Check for red-flag symptoms
            red_flags = self._detect_red_flags(user_message)
            
            # Prepare messages for Ollama
            messages = self._prepare_messages(user_message, chat_history)
            
            # Get response from Ollama
            response_data = self._call_ollama(messages)
            
            if response_data["success"]:
                # Process response into structured format
                structured_response = self._structure_response(response_data["response"])
                
                # Add red-flag warnings if detected
                if red_flags:
                    structured_response["red_flags"] = red_flags
                    structured_response["urgent_warning"] = True
                
                return {
                    "success": True,
                    "response": structured_response,
                    "red_flags": red_flags,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return response_data
                
        except Exception as e:
            logger.error(f"Error in get_medical_response: {str(e)}")
            return {
                "success": False,
                "error": f"An error occurred while processing your request: {str(e)}",
                "error_type": "processing_error"
            }
    
    def get_streaming_response(self, user_message: str, chat_history: Optional[List[Dict]] = None) -> Generator[str, None, None]:
        """Get streaming medical response from Ollama."""
        try:
            # Check if query is medical
            if not self._is_medical_query(user_message):
                yield "I can help with medical and health-related questions. Please try rephrasing your question with medical context, symptoms, conditions, or health concerns."
                return
            
            # Check for red-flag symptoms first
            red_flags = self._detect_red_flags(user_message)
            if red_flags:
                yield "🚨 **URGENT:** Based on your symptoms, you should seek immediate medical attention. Call emergency services or go to the nearest emergency room.\n\n"
            
            # Prepare messages for Ollama
            messages = self._prepare_messages(user_message, chat_history)
            
            # Stream response from Ollama with gemma2:2b - ULTRA-FAST
            payload = {
                "model": config.DEFAULT_LLM_MODEL,  # gemma2:2b
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.CHAT_TEMPERATURE,
                    "num_predict": config.DEFAULT_MAX_TOKENS,
                    "top_k": 20,
                    "top_p": 0.7,
                    "repeat_penalty": 1.0,
                    "num_ctx": 1024
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                yield chunk['message']['content']
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: Ollama service returned status code {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"An error occurred: {str(e)}"
    
    def _is_medical_query(self, message: str) -> bool:
        """Check if the query is medical/health-related using enhanced SafetyGuard."""
        # Use the enhanced SafetyGuard classification that includes snake bites and comprehensive medical vocabulary
        result = self.safety_guard.is_medical_query(message)
        return result["is_medical"]
    
    def _detect_red_flags(self, message: str) -> List[str]:
        """Detect red-flag symptoms requiring urgent medical attention."""
        message_lower = message.lower()
        detected_flags = []
        
        for symptom in config.RED_FLAG_SYMPTOMS:
            if symptom.lower() in message_lower:
                detected_flags.append(symptom)
        
        return detected_flags
    
    def _prepare_messages(self, user_message: str, chat_history: Optional[List[Dict]] = None) -> List[Dict]:
        """Prepare messages array for Ollama API."""
        messages = []
        
        # Add system prompt
        system_prompt = config.get_medical_prompts()["chat_system"]
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add chat history (last 10 messages to stay within context)
        if chat_history:
            recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        return messages
    
    def _call_ollama(self, messages: List[Dict]) -> Dict[str, Any]:
        """Make API call to Ollama using configured ultra-fast model."""

        # Use gemma2:2b model - ULTRA-FAST configuration
        payload = {
            "model": config.DEFAULT_LLM_MODEL,  # gemma2:2b
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.CHAT_TEMPERATURE,
                "num_predict": config.DEFAULT_MAX_TOKENS,
                "top_k": 20,     # Ultra-reduced for maximum speed
                "top_p": 0.7,    # Lower for faster token selection
                "repeat_penalty": 1.0,   # Minimal penalty for speed
                "num_ctx": 1024,  # Minimal context window for maximum speed
                "seed": 42  # Fixed seed for consistency
            }
        }

        logger.info(f"Using {config.DEFAULT_LLM_MODEL} model (ultra-fast)")

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if 'message' in data and 'content' in data['message']:
                    logger.info(f"Successfully received response from {config.DEFAULT_LLM_MODEL}")
                    return {
                        "success": True,
                        "response": data['message']['content'],
                        "model_used": config.DEFAULT_LLM_MODEL
                    }
                else:
                    return {
                        "success": False,
                        "error": "Invalid response format from Ollama",
                        "error_type": "invalid_response"
                    }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "error_type": "http_error"
                }

        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out - try a shorter question",
                "error_type": "timeout"
            }
        except requests.exceptions.ConnectionError:
            return {
                "success": False,
                "error": "Cannot connect to Ollama service - check if it's running",
                "error_type": "connection_error"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "error_type": "unexpected_error"
            }
    
    def _structure_response(self, raw_response: str) -> Dict[str, Any]:
        """Structure the raw Ollama response - SIMPLIFIED FOR GUARANTEED DISPLAY."""
        # ALWAYS return the full response in summary - NO MORE PARSING FAILURES!
        structured = {
            "summary": raw_response,  # ALWAYS show the full response
            "possible_causes": "",
            "what_to_do_now": "",
            "what_to_avoid": "",
            "when_to_seek_urgent_care": "",
            "questions_for_clinician": "",
            "confidence": "AI medical advice - consult healthcare provider for diagnosis",
            "raw_response": raw_response
        }

        # Simple extraction without complex parsing
        try:
            # Extract likely causes section
            if "**Likely Causes:**" in raw_response:
                causes_start = raw_response.find("**Likely Causes:**")
                causes_end = raw_response.find("**What To Do:**", causes_start)
                if causes_end > causes_start:
                    structured["possible_causes"] = raw_response[causes_start:causes_end].replace("**Likely Causes:**", "").strip()

            # Extract what to do section
            if "**What To Do:**" in raw_response:
                todo_start = raw_response.find("**What To Do:**")
                todo_end = raw_response.find("**See Doctor If:**", todo_start)
                if todo_end > todo_start:
                    structured["what_to_do_now"] = raw_response[todo_start:todo_end].replace("**What To Do:**", "").strip()

            # Extract urgent care section
            if "**See Doctor If:**" in raw_response:
                urgent_start = raw_response.find("**See Doctor If:**")
                structured["when_to_seek_urgent_care"] = raw_response[urgent_start:].replace("**See Doctor If:**", "").strip()

        except Exception as e:
            # Even if parsing fails, we still have the full response in summary
            logger.info(f"Section parsing failed but full response available: {e}")

        return structured
#!/usr/bin/env python3
"""
MediLens Local - Chat Service
Core Ollama integration and medical chat functionality.
"""

import requests
import json
import time
import re
import platform
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
        # Initialize with universal compatibility
        self.base_url = config.OLLAMA_BASE_URL
        self.timeout = config.get_os_optimized_timeout(config.OLLAMA_TIMEOUT)
        self.retry_attempts = config.OLLAMA_RETRY_ATTEMPTS
        self.model = config.DEFAULT_LLM_MODEL
        self.safety_guard = SafetyGuard()

        # Initialize connection and ensure models
        self._initialize_universal_connection()
        
    def _initialize_universal_connection(self) -> None:
        """Initialize universal cross-platform Ollama connection."""
        try:
            # Step 1: Detect or start Ollama service
            is_running, url, message = config.detect_ollama_service()

            if not is_running:
                logger.info("Ollama not detected, attempting to start...")
                started, start_message = config.start_ollama_service()
                if started:
                    is_running, url, message = config.detect_ollama_service()

            if is_running:
                self.base_url = url  # Use detected working URL
                logger.info(f"Ollama connection established: {message}")

                # Step 2: Ensure models are available
                success, available, missing = config.ensure_models_available()
                if success and available:
                    # Use best available model
                    best_model = config.get_best_available_model()
                    if best_model:
                        self.model = best_model
                        logger.info(f"Using model: {self.model}")

            else:
                logger.warning(f"Failed to initialize Ollama: {message}")

        except Exception as e:
            logger.error(f"Error in universal connection init: {str(e)}")

    def test_connection(self) -> Tuple[bool, str]:
        """Universal connection test with cross-platform compatibility."""
        try:
            # Try current base_url first
            response = requests.get(f"{self.base_url}/api/tags",
                                  timeout=config.OLLAMA_CONNECTION_TIMEOUT)

            if response.status_code == 200:
                data = response.json()
                installed_models = [model['name'] for model in data.get('models', [])]

                # Check for any available models from hierarchy
                available_from_hierarchy = []
                for model in config.MODEL_HIERARCHY:
                    if any(model in installed for installed in installed_models):
                        available_from_hierarchy.append(model)

                if available_from_hierarchy:
                    # Use best available model
                    self.model = available_from_hierarchy[0]
                    system_info = platform.system()
                    return True, f"✅ Connected to Ollama on {system_info} with {self.model}"
                else:
                    return False, f"❌ No compatible models found. Available: {installed_models}"
            else:
                return False, f"❌ Ollama service error: {response.status_code}"

        except requests.exceptions.ConnectionError:
            # Try to detect service on alternative ports/URLs
            is_running, url, message = config.detect_ollama_service()
            if is_running:
                self.base_url = url
                return self.test_connection()  # Retry with new URL
            else:
                return False, "❌ Cannot connect to Ollama. Service may not be running."
        except requests.exceptions.Timeout:
            return False, "❌ Connection timeout. Ollama may be starting up."
        except Exception as e:
            return False, f"❌ Connection error: {str(e)}"
    
    
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
            
            # Universal streaming with current best model
            payload = {
                "model": self.model,
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
        """Universal cross-platform Ollama API call with robust error handling."""

        # Use current best model with OS-optimized settings
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.CHAT_TEMPERATURE,
                "num_predict": config.DEFAULT_MAX_TOKENS,
                "top_k": 20,
                "top_p": 0.7,
                "repeat_penalty": 1.0,
                "num_ctx": 1024,
                "seed": 42
            }
        }

        logger.info(f"Using {self.model} on {platform.system()}")

        # Retry with progressive fallback
        for attempt in range(self.retry_attempts):
            try:
                timeout = self.timeout * (1 + attempt * 0.5)  # Progressive timeout increase

                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=timeout
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'message' in data and 'content' in data['message']:
                        logger.info(f"✅ Response received from {self.model} (attempt {attempt + 1})")
                        return {
                            "success": True,
                            "response": data['message']['content'],
                            "model_used": self.model,
                            "attempt": attempt + 1
                        }
                    else:
                        logger.warning(f"Invalid response format (attempt {attempt + 1})")
                        continue

                elif response.status_code == 404:
                    # Model not found - try fallback
                    logger.warning(f"Model {self.model} not found, trying fallback...")
                    if self._try_fallback_model():
                        payload["model"] = self.model
                        continue
                    else:
                        return {
                            "success": False,
                            "error": f"Model {self.model} not available and no fallbacks found",
                            "error_type": "model_not_found"
                        }
                else:
                    logger.warning(f"HTTP {response.status_code} (attempt {attempt + 1})")
                    if attempt == self.retry_attempts - 1:  # Last attempt
                        return {
                            "success": False,
                            "error": f"HTTP {response.status_code}: {response.text}",
                            "error_type": "http_error"
                        }

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1}/{self.retry_attempts}")
                if attempt == self.retry_attempts - 1:
                    return {
                        "success": False,
                        "error": f"Request timed out after {self.retry_attempts} attempts. Try a shorter question.",
                        "error_type": "timeout"
                    }

            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error on attempt {attempt + 1}")
                # Try to reconnect
                if self._attempt_reconnection():
                    continue
                elif attempt == self.retry_attempts - 1:
                    return {
                        "success": False,
                        "error": "Cannot connect to Ollama service. Please check if Ollama is running.",
                        "error_type": "connection_error"
                    }

            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.retry_attempts - 1:
                    return {
                        "success": False,
                        "error": f"Unexpected error: {str(e)}",
                        "error_type": "unexpected_error"
                    }

            # Brief delay between retries
            if attempt < self.retry_attempts - 1:
                time.sleep(1 + attempt)

        return {
            "success": False,
            "error": "All retry attempts failed",
            "error_type": "retry_exhausted"
        }

    def _try_fallback_model(self) -> bool:
        """Try to switch to a fallback model."""
        try:
            best_model = config.get_best_available_model()
            if best_model and best_model != self.model:
                logger.info(f"Switching from {self.model} to {best_model}")
                self.model = best_model
                return True
            return False
        except:
            return False

    def _attempt_reconnection(self) -> bool:
        """Attempt to reconnect to Ollama service."""
        try:
            is_running, url, message = config.detect_ollama_service()
            if is_running:
                self.base_url = url
                logger.info(f"Reconnected to Ollama at {url}")
                return True
            return False
        except:
            return False
    
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
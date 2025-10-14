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
from services.medical_safety import medical_safety
from services.medical_knowledge import medical_knowledge
from services.clinical_validation import clinical_validator
from services.legal_compliance import legal_compliance

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
                # CRITICAL MEDICAL SAFETY VALIDATION
                is_safe, safe_response = medical_safety.validate_medical_response(
                    user_message, response_data["response"]
                )

                if not is_safe:
                    # Response blocked for safety - return safety message
                    return {
                        "success": True,
                        "response": {"summary": safe_response, "safety_blocked": True},
                        "safety_intervention": True,
                        "timestamp": datetime.now().isoformat()
                    }

                # MEDICAL KNOWLEDGE ENHANCEMENT
                knowledge_enhancement = medical_knowledge.enhance_medical_response(
                    user_message, safe_response
                )

                # Use enhanced response with medical validation
                final_response = knowledge_enhancement["enhanced_response"]

                # CLINICAL VALIDATION AND OVERSIGHT
                clinical_validation = clinical_validator.validate_clinical_response(
                    user_message, final_response
                )

                # Process enhanced response into structured format
                structured_response = self._structure_response(final_response)

                # Add medical enhancement metadata
                structured_response["medical_accuracy_score"] = knowledge_enhancement.get("accuracy_score", 0.7)
                structured_response["specialist_referral"] = knowledge_enhancement.get("specialist_referral")
                structured_response["clinical_context"] = knowledge_enhancement.get("clinical_context")

                # Add clinical validation metadata
                structured_response["clinical_validation_level"] = clinical_validation.validation_level.value
                structured_response["clinical_accuracy_score"] = clinical_validation.clinical_accuracy_score
                structured_response["evidence_quality"] = clinical_validation.evidence_quality
                structured_response["clinical_notes"] = clinical_validation.clinical_notes
                structured_response["contraindications"] = clinical_validation.contraindications
                structured_response["liability_risk"] = clinical_validation.liability_risk

                # LEGAL COMPLIANCE AND LIABILITY PROTECTION
                safety_dict = {
                    "safety_level": response_data.get("safety_level", "safe"),
                    "emergency_detected": bool(red_flags),
                    "urgent_detected": False,
                    "blocked_response": False
                }

                clinical_dict = {
                    "validation_level": clinical_validation.validation_level.value,
                    "clinical_accuracy_score": clinical_validation.clinical_accuracy_score
                }

                compliance_record = legal_compliance.assess_legal_compliance(
                    user_message, final_response, safety_dict, clinical_dict
                )

                # Add legal compliance metadata
                structured_response["compliance_level"] = compliance_record.compliance_level.value
                structured_response["legal_liability_risk"] = compliance_record.liability_risk.value
                structured_response["disclaimers_present"] = compliance_record.disclaimers_present
                structured_response["professional_supervision_required"] = compliance_record.professional_supervision
                structured_response["legal_recommendations"] = compliance_record.legal_recommendations
                structured_response["compliance_record_id"] = compliance_record.record_id

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
        """Get streaming medical response from Ollama with critical safety validation."""
        try:
            # CRITICAL: Emergency safety check FIRST
            safety_analysis = medical_safety.analyze_medical_safety(user_message, "")

            if safety_analysis["emergency_detected"]:
                # EMERGENCY: Block streaming and show immediate safety message
                yield medical_safety.critical_disclaimer
                return

            if safety_analysis["urgent_detected"]:
                # URGENT: Show warning but allow response with disclaimer
                yield medical_safety.urgent_disclaimer + "\n\n---\n\n"

            # Check if query is medical
            if not self._is_medical_query(user_message):
                yield "I can help with medical and health-related questions. Please try rephrasing your question with medical context, symptoms, conditions, or health concerns."
                return

            # Check for red-flag symptoms (legacy support)
            red_flags = self._detect_red_flags(user_message)
            if red_flags and not safety_analysis["emergency_detected"]:
                yield "⚠️ **IMPORTANT:** Based on your symptoms, consider consulting a healthcare provider promptly.\n\n"
            
            # Prepare messages for Ollama
            messages = self._prepare_messages(user_message, chat_history)
            
            # ACCURACY-OPTIMIZED for meditron:7b (medical expert model)
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": config.CHAT_TEMPERATURE,  # 0.3 for balanced accuracy
                    "num_predict": config.DEFAULT_MAX_TOKENS,  # 600 for complete responses
                    "top_k": config.TOP_K,  # 40 for natural language
                    "top_p": config.TOP_P,  # 0.9 for comprehensive answers
                    "repeat_penalty": config.REPEAT_PENALTY,  # 1.1 gentle
                    "num_ctx": config.NUM_CTX  # 2048 for better reasoning
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                response_text = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            if 'message' in chunk and 'content' in chunk['message']:
                                content = chunk['message']['content']
                                response_text += content
                                yield content
                        except json.JSONDecodeError:
                            continue

                # NO DISCLAIMER - Keep response clean and concise
                pass
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

        # ACCURACY-OPTIMIZED for meditron:7b - HIGH-QUALITY medical responses
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": config.CHAT_TEMPERATURE,  # 0.3 for balanced accuracy
                "num_predict": config.DEFAULT_MAX_TOKENS,  # 600 for complete responses
                "top_k": config.TOP_K,  # 40 for natural language
                "top_p": config.TOP_P,  # 0.9 for comprehensive answers
                "repeat_penalty": config.REPEAT_PENALTY,  # 1.1 gentle
                "num_ctx": config.NUM_CTX,  # 2048 for better medical reasoning
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
#!/usr/bin/env python3
"""
MediLens Local - Vision Service
Medical image analysis using llava:7b vision model.
"""

import io
import base64
import requests
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import logging
import pytesseract

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionAnalyzer:
    """Handles medical image analysis using llava:7b vision model."""
    
    def __init__(self):
        self.base_url = config.OLLAMA_BASE_URL
        self.model = config.DEFAULT_VISION_MODEL
        self.timeout = config.OLLAMA_VISION_TIMEOUT  # Optimized timeout for faster vision analysis
        
    def test_vision_model_connection(self) -> Tuple[bool, str]:
        """Test connection to vision model."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                if self.model in models or any(self.model in model for model in models):
                    return True, f"Vision model {self.model} is available"
                else:
                    return False, f"Vision model {self.model} not found. Available models: {models}"
            else:
                return False, f"Ollama service returned status code: {response.status_code}"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def validate_medical_image(self, image_bytes: bytes) -> Tuple[bool, str]:
        """Validate if image is suitable for medical analysis."""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return False, "Invalid image format"
            
            # Check image dimensions
            height, width = image.shape[:2]
            if height < 100 or width < 100:
                return False, "Image too small for analysis (minimum 100x100 pixels)"
            
            if height > 4000 or width > 4000:
                return False, "Image too large (maximum 4000x4000 pixels)"
            
            # Check if image is too dark (more lenient brightness validation)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray)
            
            if mean_brightness < 10:  # Very dark threshold
                return False, "Image too dark for analysis"
            # Removed overly restrictive brightness check - medical images can be bright
            
            return True, "Image suitable for medical analysis"
            
        except Exception as e:
            logger.error(f"Error validating image: {e}")
            return False, f"Image validation failed: {str(e)}"
    
    def preprocess_medical_image(self, image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """Preprocess medical image for better analysis."""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Get original dimensions
            original_height, original_width = image.shape[:2]

            # Resize if too large (keep aspect ratio)
            max_dimension = 1024
            if max(original_height, original_width) > max_dimension:
                if original_height > original_width:
                    new_height = max_dimension
                    new_width = int((original_width * max_dimension) / original_height)
                else:
                    new_width = max_dimension
                    new_height = int((original_height * max_dimension) / original_width)

                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Enhance image quality
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)

            # Blend with original (subtle enhancement)
            enhanced = cv2.addWeighted(image, 0.7, sharpened, 0.3, 0)

            # Convert back to bytes
            _, buffer = cv2.imencode('.jpg', enhanced, [cv2.IMWRITE_JPEG_QUALITY, 90])
            processed_bytes = buffer.tobytes()

            # Prepare metadata
            metadata = {
                "original_size": f"{original_width}x{original_height}",
                "processed_size": f"{enhanced.shape[1]}x{enhanced.shape[0]}",
                "enhancement_applied": "Subtle sharpening and quality optimization",
                "compression_quality": 90
            }

            return processed_bytes, metadata

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image_bytes, {"error": str(e)}

    def preprocess_medical_image_fast(self, image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """SPEED-OPTIMIZED: Fast image preprocessing for quicker analysis."""
        try:
            # Convert bytes to image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Get original dimensions
            original_height, original_width = image.shape[:2]

            # SPEED: More aggressive resizing for faster processing
            max_dimension = 512  # Smaller size for speed
            if max(original_height, original_width) > max_dimension:
                if original_height > original_width:
                    new_height = max_dimension
                    new_width = int((original_width * max_dimension) / original_height)
                else:
                    new_width = max_dimension
                    new_height = int((original_height * max_dimension) / original_width)

                # Use faster interpolation
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            # SPEED: Skip enhancement for faster processing
            # Convert back to bytes with faster compression
            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 70])  # Lower quality for speed
            processed_bytes = buffer.tobytes()

            # Minimal metadata
            metadata = {
                "original_size": f"{original_width}x{original_height}",
                "processed_size": f"{image.shape[1]}x{image.shape[0]}",
                "optimization": "Fast processing - reduced quality for speed",
                "compression_quality": 70
            }

            return processed_bytes, metadata

        except Exception as e:
            logger.error(f"Error in fast preprocessing: {e}")
            return image_bytes, {"error": str(e)}
    
    def encode_image_for_llava(self, image_bytes: bytes) -> str:
        """Encode image to base64 for llava model."""
        try:
            return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            return ""
    
    def extract_text_from_image(self, image_bytes: bytes) -> Tuple[str, float]:
        """Extract text from medical images using OCR."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return "", 0.0
            
            # Try multiple OCR approaches for medical images
            extraction_results = []
            
            # Method 1: Standard OCR
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                pil_image = Image.fromarray(gray)
                text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 6')
                if text.strip():
                    extraction_results.append((text.strip(), 0.8, "standard"))
            except Exception as e:
                logger.warning(f"Standard OCR failed: {e}")
            
            # Method 2: High contrast for medical text
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.equalizeHist(gray)
                _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                pil_enhanced = Image.fromarray(binary)
                enhanced_text = pytesseract.image_to_string(pil_enhanced, config='--oem 3 --psm 6')
                if enhanced_text.strip():
                    extraction_results.append((enhanced_text.strip(), 0.7, "enhanced"))
            except Exception as e:
                logger.warning(f"Enhanced OCR failed: {e}")
            
            # Method 3: Document mode for prescriptions/reports
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                pil_image = Image.fromarray(gray)
                doc_text = pytesseract.image_to_string(pil_image, config='--oem 3 --psm 4')
                if doc_text.strip():
                    extraction_results.append((doc_text.strip(), 0.6, "document"))
            except Exception as e:
                logger.warning(f"Document OCR failed: {e}")
            
            # Select best result
            if extraction_results:
                best_result = max(extraction_results, key=lambda x: len(x[0]) * x[1])
                return best_result[0], best_result[1]
            
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return "", 0.0
    
    def analyze_medical_image(self, image_bytes: bytes, user_query: str = None) -> Dict[str, Any]:
        """Analyze medical image using llava vision model - SPEED OPTIMIZED."""
        try:
            # Quick image validation (optimized)
            is_valid, validation_msg = self.validate_medical_image(image_bytes)
            if not is_valid:
                return {
                    "success": False,
                    "error": validation_msg,
                    "error_type": "invalid_image"
                }

            # SPEED OPTIMIZATION: Skip OCR for faster analysis unless specifically needed
            extracted_text = ""
            ocr_confidence = 0.0

            # Only run OCR if user specifically asks about text or document
            if user_query and any(keyword in user_query.lower() for keyword in ['text', 'read', 'document', 'report', 'lab', 'result']):
                extracted_text, ocr_confidence = self.extract_text_from_image(image_bytes)

            # Fast image preprocessing (reduced quality for speed)
            processed_bytes, metadata = self.preprocess_medical_image_fast(image_bytes)

            # Encode for vision model
            base64_image = self.encode_image_for_llava(processed_bytes)
            if not base64_image:
                return {
                    "success": False,
                    "error": "Failed to encode image",
                    "error_type": "encoding_error"
                }

            # Use standardized medical vision prompt from config
            vision_system_prompt = config.get_medical_prompts()["vision_system"]

            if extracted_text.strip():
                context_prompt = f"""{vision_system_prompt}

ADDITIONAL CONTEXT FROM OCR:
{extracted_text[:300]}

USER REQUEST: {user_query if user_query else "Analyze this medical image for conditions and provide care guidance."}"""
            else:
                context_prompt = f"""{vision_system_prompt}

USER REQUEST: {user_query if user_query else "Analyze this medical image for conditions and provide care guidance."}"""
            
            # SPEED-OPTIMIZED: Prepare request for faster llava model response
            payload = {
                "model": self.model,
                "prompt": context_prompt,  # Use direct prompt without system prefix for speed
                "images": [base64_image],
                "stream": False,
                "options": {
                    "temperature": 0.2,  # Lower for faster, more focused responses
                    "num_predict": 250,   # Reduced token limit for speed
                    "top_p": 0.9,        # Speed optimization
                    "top_k": 50,         # Speed optimization
                    "repeat_penalty": 1.1  # Prevent repetition for efficiency
                }
            }
            
            # Try available models for speed and reliability  
            last_error = None
            available_models = [config.DEFAULT_VISION_MODEL]  # Vision models limited to llava for now
            
            for model_attempt, model_to_try in enumerate(available_models):
                # Check if vision model is available
                try:
                    test_response = requests.get(f"{self.base_url}/api/tags", timeout=3)
                    if test_response.status_code == 200:
                        data = test_response.json()
                        installed_models = [model['name'] for model in data.get('models', [])]
                        if not any(model_to_try in installed for installed in installed_models):
                            continue  # Skip this model if not installed
                    else:
                        continue
                except:
                    continue
                
                # Update payload with current model
                payload["model"] = model_to_try  # Note: vision uses different API format
                logger.info(f"Vision analysis with model: {model_to_try}")
            
                for attempt in range(config.OLLAMA_RETRY_ATTEMPTS):
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/generate",
                            json=payload,
                            timeout=self.timeout
                        )

                        if response.status_code == 200:
                            data = response.json()
                            analysis_text = data.get('response', '')

                            if analysis_text:
                                # Structure the response
                                structured_analysis = self.structure_vision_response(analysis_text)

                                return {
                                    "success": True,
                                    "analysis": structured_analysis,
                                    "raw_response": analysis_text,
                                    "image_metadata": metadata,
                                    "model_used": model_to_try,
                                    "timestamp": config.get_timestamp(),
                                    "ocr_extracted_text": extracted_text,
                                    "ocr_confidence": ocr_confidence,
                                    "analysis_mode": "dual" if extracted_text.strip() else "visual",
                                    "attempts": attempt + 1
                                }
                            else:
                                last_error = "Empty response from vision model"
                        else:
                            last_error = f"HTTP {response.status_code}: {response.text}"

                    except requests.exceptions.Timeout:
                        last_error = f"Vision analysis timed out after {self.timeout} seconds"
                        logger.warning(f"Vision analysis timeout on attempt {attempt + 1}")
                    except requests.exceptions.ConnectionError:
                        last_error = "Cannot connect to Ollama service"
                        logger.warning(f"Connection error on attempt {attempt + 1}")
                    except Exception as e:
                        last_error = f"Unexpected error: {str(e)}"
                        logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

                    # Fast retry for vision analysis
                    if attempt < config.OLLAMA_RETRY_ATTEMPTS - 1:
                        time.sleep(0.8 + attempt * 0.4)  # Quick retry: 0.8s, 1.2s
                
                logger.warning(f"Vision model {model_to_try} failed")
            
            # All attempts failed
            return {
                "success": False,
                "error": f"Vision analysis failed after {config.OLLAMA_RETRY_ATTEMPTS} attempts. Last error: {last_error}",
                "error_type": "connection_failed"
            }
                
        except Exception as e:
            logger.error(f"Error in medical image analysis: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "error_type": "analysis_error"
            }
    
    def get_medical_vision_prompt(self) -> str:
        """Get specialized prompt for medical image analysis."""
        return """You are a medical image analysis AI assistant. Analyze images conservatively.

CRITICAL GUIDELINES:
- Only analyze clearly medical-related images (wounds, rashes, medical conditions)
- If image shows normal appearance or non-medical content, state so clearly
- Be conservative - avoid emergency language unless truly warranted
- This is preliminary observation only, NOT diagnosis

RESPONSE FORMAT (5-8 lines):
1. What you observe in the image
2. Assessment of any visible medical concerns
3. General care recommendations (if applicable)  
4. When to seek professional evaluation

AVOID: Emergency language unless genuinely life-threatening conditions are visible."""

    def get_enhanced_medical_vision_prompt(self) -> str:
        """Get optimized medical vision prompt from config."""
        return config.get_medical_prompts()["vision_system"]
    
    def structure_vision_response(self, raw_response: str) -> Dict[str, str]:
        """Structure the raw vision response into organized sections."""
        try:
            # Simple structured response
            structured = {
                "observation": "",
                "concerns": "", 
                "immediate_care": "",
                "seek_care_if": "",
                "full_response": raw_response.strip()
            }
            
            # Try to extract key sections from response
            lines = raw_response.strip().split('\n')
            current_content = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_content.append(line)
            
            # If response is structured, try to parse sections
            if len(current_content) >= 3:
                structured["observation"] = current_content[0]
                if len(current_content) >= 4:
                    structured["concerns"] = current_content[1]
                    structured["immediate_care"] = current_content[2]
                    structured["seek_care_if"] = " ".join(current_content[3:])
                else:
                    structured["immediate_care"] = " ".join(current_content[1:])
            else:
                # Fallback to full response
                structured["observation"] = raw_response.strip()
            
            return structured
            
        except Exception as e:
            logger.error(f"Error structuring vision response: {e}")
            return {
                "observation": raw_response.strip(),
                "concerns": "",
                "immediate_care": "",
                "seek_care_if": "",
                "full_response": raw_response.strip()
            }
    
    def get_emergency_keywords(self) -> List[str]:
        """Get list of keywords that trigger emergency warnings - more specific to avoid false positives."""
        return [
            "call 911", "call emergency", "emergency services", "ambulance",
            "life-threatening", "critical condition", "immediate danger",
            "poisoning", "overdose", "unconscious", "not breathing",
            "severe chest pain", "heart attack", "stroke symptoms",
            "severe bleeding", "major bleeding", "severe head injury",
            "compound fracture", "severe burn"
        ]
    
    def detect_emergency_indicators(self, analysis_text: str) -> List[str]:
        """Detect emergency indicators in analysis."""
        try:
            emergency_keywords = self.get_emergency_keywords()
            detected = []
            
            analysis_lower = analysis_text.lower()
            for keyword in emergency_keywords:
                if keyword in analysis_lower:
                    detected.append(keyword)
            
            return detected
            
        except Exception as e:
            logger.error(f"Error detecting emergency indicators: {e}")
            return []
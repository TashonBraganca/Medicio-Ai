#!/usr/bin/env python3
"""
MediLens Local - Document Processing Service
OCR integration and medical document analysis.
"""

import os
import io
import cv2
import pytesseract
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import PyPDF2
import logging
import requests

from config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles OCR, document processing, and medical analysis."""
    
    def __init__(self):
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/()%+= '
        self.setup_tesseract()

        # Medical value thresholds for highlighting
        self.critical_values = {
            'glucose': {'high': 180, 'low': 70, 'units': ['mg/dl', 'mg/dL']},
            'cholesterol': {'high': 240, 'low': 0, 'units': ['mg/dl', 'mg/dL']},
            'blood_pressure': {'systolic_high': 140, 'diastolic_high': 90},
            'hemoglobin': {'high': 16, 'low': 12, 'units': ['g/dl', 'g/dL']},
            'creatinine': {'high': 1.2, 'low': 0, 'units': ['mg/dl', 'mg/dL']},
            'bun': {'high': 20, 'low': 0, 'units': ['mg/dl', 'mg/dL']},
            'heart_rate': {'high': 100, 'low': 60, 'units': ['bpm']},
            'temperature': {'high': 99.5, 'low': 97, 'units': ['f', '°f']}
        }
    
    def setup_tesseract(self) -> bool:
        """Setup Tesseract OCR with automatic path detection."""
        try:
            # Common Windows Tesseract paths
            possible_paths = [
                r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                r'C:\Users\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
                'tesseract'  # If in PATH
            ]
            
            for path in possible_paths:
                try:
                    if path == 'tesseract':
                        # Test if in PATH
                        result = os.system('tesseract --version > nul 2>&1')
                        if result == 0:
                            logger.info("Tesseract found in system PATH")
                            return True
                    else:
                        if os.path.exists(path):
                            pytesseract.pytesseract.tesseract_cmd = path
                            logger.info(f"Tesseract found at: {path}")
                            return True
                except Exception as e:
                    continue
            
            logger.warning("Tesseract OCR not found. Document processing will be limited.")
            return False
            
        except Exception as e:
            logger.error(f"Error setting up Tesseract: {e}")
            return False

    def highlight_critical_values(self, text: str) -> str:
        """Highlight critical medical values in text with color coding."""
        try:
            highlighted_text = text

            # Define color codes for different severities
            colors = {
                'critical_high': '🔴 **CRITICAL HIGH**',
                'high': '🟠 **HIGH**',
                'low': '🟡 **LOW**',
                'critical_low': '🔴 **CRITICAL LOW**',
                'normal': '🟢 **NORMAL**'
            }

            # Blood pressure pattern (e.g., "120/80")
            bp_pattern = r'(?:blood pressure|bp)[:\s]*(\d{2,3})/(\d{2,3})'
            bp_matches = re.finditer(bp_pattern, highlighted_text, re.IGNORECASE)
            for match in bp_matches:
                systolic = int(match.group(1))
                diastolic = int(match.group(2))
                original = match.group(0)

                if systolic >= 180 or diastolic >= 110:
                    highlighted_text = highlighted_text.replace(original, f"{colors['critical_high']} {original}")
                elif systolic >= 140 or diastolic >= 90:
                    highlighted_text = highlighted_text.replace(original, f"{colors['high']} {original}")
                else:
                    highlighted_text = highlighted_text.replace(original, f"{colors['normal']} {original}")

            # General lab values (e.g., "Glucose: 180 mg/dL")
            for test_name, thresholds in self.critical_values.items():
                if test_name == 'blood_pressure':
                    continue  # Already handled above

                # Create pattern for this test
                units_pattern = '|'.join(thresholds.get('units', ['']))
                pattern = rf'(?:{test_name})[:\s]*(\d+\.?\d*)\s*(?:{units_pattern})?'

                matches = re.finditer(pattern, highlighted_text, re.IGNORECASE)
                for match in matches:
                    value = float(match.group(1))
                    original = match.group(0)

                    # Determine severity and highlight
                    if 'high' in thresholds and value > thresholds['high']:
                        if value > thresholds['high'] * 1.5:  # Critical threshold
                            highlighted_text = highlighted_text.replace(original, f"{colors['critical_high']} {original}")
                        else:
                            highlighted_text = highlighted_text.replace(original, f"{colors['high']} {original}")
                    elif 'low' in thresholds and value < thresholds['low']:
                        if value < thresholds['low'] * 0.7:  # Critical low threshold
                            highlighted_text = highlighted_text.replace(original, f"{colors['critical_low']} {original}")
                        else:
                            highlighted_text = highlighted_text.replace(original, f"{colors['low']} {original}")
                    else:
                        highlighted_text = highlighted_text.replace(original, f"{colors['normal']} {original}")

            return highlighted_text

        except Exception as e:
            logger.error(f"Error highlighting critical values: {e}")
            return text
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced medical document preprocessing for superior OCR accuracy."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            # Method 2: Advanced denoising for medical documents
            denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)

            # Method 3: Adaptive thresholding for better text extraction
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # Method 4: Morphological operations to clean medical document artifacts
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Additional sharpening for medical text
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(processed, -1, kernel_sharp)

            return sharpened

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def extract_text_from_image(self, image_bytes: bytes) -> Tuple[str, float]:
        """Extract complete text from medical documents using specialized OCR methods."""
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return "", 0.0
            
            # Multiple specialized OCR approaches for medical documents
            extraction_results = []
            
            # Method 1: Medical document optimized preprocessing
            try:
                # Enhanced preprocessing for medical lab reports
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Noise reduction and contrast enhancement
                denoised = cv2.medianBlur(gray, 3)
                enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(denoised)
                
                # Adaptive thresholding for better text detection
                binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, 2)
                
                pil_image = Image.fromarray(binary)
                
                # Use comprehensive OCR config for medical documents
                medical_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,:-/()%+ '
                medical_text = pytesseract.image_to_string(pil_image, config=medical_config)
                
                if medical_text.strip():
                    extraction_results.append((medical_text.strip(), 0.9, "medical_optimized"))
                    
            except Exception as e:
                logger.warning(f"Medical OCR failed: {e}")
            
            # Method 2: Table detection mode for lab results
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # High contrast binary for table detection
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                pil_binary = Image.fromarray(binary)
                
                # PSM 6 for uniform text blocks (good for medical tables)
                table_text = pytesseract.image_to_string(pil_binary, config='--oem 3 --psm 6')
                if table_text.strip():
                    extraction_results.append((table_text.strip(), 0.8, "table_detection"))
                    
            except Exception as e:
                logger.warning(f"Table OCR failed: {e}")
            
            # Method 3: Conservative approach with original image
            try:
                pil_original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                conservative_text = pytesseract.image_to_string(pil_original, config='--oem 3 --psm 4')
                if conservative_text.strip():
                    extraction_results.append((conservative_text.strip(), 0.7, "conservative"))
                    
            except Exception as e:
                logger.warning(f"Conservative OCR failed: {e}")
            
            # Method 4: High-resolution upscaling for small text
            try:
                # Upscale image for better OCR of small text
                height, width = image.shape[:2]
                upscaled = cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
                gray_up = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
                
                # Enhanced processing for upscaled image
                enhanced_up = cv2.equalizeHist(gray_up)
                _, binary_up = cv2.threshold(enhanced_up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                pil_up = Image.fromarray(binary_up)
                
                upscaled_text = pytesseract.image_to_string(pil_up, config='--oem 3 --psm 6')
                if upscaled_text.strip():
                    extraction_results.append((upscaled_text.strip(), 0.75, "upscaled"))
                    
            except Exception as e:
                logger.warning(f"Upscaled OCR failed: {e}")
            
            # Combine and select best results
            if extraction_results:
                # Sort by confidence * length to get most complete result
                best_result = max(extraction_results, key=lambda x: len(x[0]) * x[1])
                
                # Additional validation - ensure we have key medical data
                text = best_result[0]
                confidence = best_result[1]
                
                # Log extraction method for debugging
                logger.info(f"Best OCR method: {best_result[2]}, confidence: {confidence:.2f}, text length: {len(text)}")
                
                return text, confidence
            
            return "", 0.0
            
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return f"OCR Error: {str(e)}", 0.0
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF document."""
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text_content = []
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(f"--- Page {page_num + 1} ---")
                        text_content.append(page_text.strip())
                except Exception as e:
                    logger.warning(f"Error extracting from page {page_num + 1}: {e}")
                    continue
            
            return "\n\n".join(text_content)
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return f"PDF Error: {str(e)}"
    
    def process_document(self, file_bytes: bytes, file_type: str, filename: str) -> Dict[str, Any]:
        """Process uploaded document and extract text."""
        try:
            extracted_text = ""
            confidence = 1.0
            processing_method = ""
            
            if file_type == "application/pdf":
                extracted_text = self.extract_text_from_pdf(file_bytes)
                processing_method = "PDF text extraction"
                
            elif file_type.startswith("image/"):
                extracted_text, confidence = self.extract_text_from_image(file_bytes)
                processing_method = f"OCR (confidence: {confidence:.2%})"
                
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_type}",
                    "error_type": "unsupported_format"
                }
            
            if not extracted_text or extracted_text.startswith("OCR Error") or extracted_text.startswith("PDF Error"):
                return {
                    "success": False,
                    "error": extracted_text if extracted_text else "No text could be extracted",
                    "error_type": "extraction_failed"
                }
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "confidence": confidence,
                "processing_method": processing_method,
                "filename": filename,
                "file_type": file_type,
                "text_length": len(extracted_text),
                "word_count": len(extracted_text.split()) if extracted_text else 0
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {
                "success": False,
                "error": f"Document processing failed: {str(e)}",
                "error_type": "processing_error"
            }
    
    def analyze_medical_document(self, extracted_text: str, ollama_base_url: str = None) -> Dict[str, Any]:
        """Analyze medical document using context-driven LLM prompts."""
        try:
            if not ollama_base_url:
                ollama_base_url = config.OLLAMA_BASE_URL
            
            # Use simplified prompt from config for speed
            system_prompt = config.get_medical_prompts()["ocr_system"]
            
            # SPEED-OPTIMIZED: Concise prompt for faster analysis
            user_prompt = f"""MEDICAL DOCUMENT TEXT:
{extracted_text[:1000]}{'...' if len(extracted_text) > 1000 else ''}

TASK: Quick analysis focusing on:
1. Document type and key findings
2. Critical values and their significance
3. Immediate medical recommendations

Analyze the above medical text efficiently. 

Provide a comprehensive medical interpretation following your system format:
- Identify the document type based on the content
- Extract all lab values, medications, and medical findings with their specific numbers/dosages
- Interpret what these values mean clinically (compare to normal ranges where applicable) 
- Provide specific medical recommendations based on the findings
- Include relevant questions for the patient to ask their healthcare provider

Base your entire analysis on the actual content extracted from this document. Quote specific values and information from the text above."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # SPEED-OPTIMIZED: Ultra-fast document analysis configuration
            available_models = config.get_available_models()

            # Use gemma2:2b for ultra-fast, high-quality medical analysis
            payload = {
                "model": config.DEFAULT_LLM_MODEL,  # gemma2:2b - ultra-fast and accurate
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.1,      # Very focused for medical accuracy
                    "num_predict": 400,      # Sufficient tokens for comprehensive analysis
                    "top_p": 0.7,           # Focused responses
                    "top_k": 20,            # Maximum speed optimization
                    "repeat_penalty": 1.1,  # Prevent repetition
                    "num_ctx": 1024,        # Minimal context for speed
                    "seed": 42              # Consistency
                }
            }
            
            # Use ultra-fast model priority: gemma2:2b > qwen2:1.5b > fallback
            fast_models = [config.DEFAULT_LLM_MODEL, config.FALLBACK_LLM_MODEL]
            last_error = None

            for model_to_try in fast_models:
                # Check if model is available
                try:
                    test_response = requests.get(f"{ollama_base_url}/api/tags", timeout=3)
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
                payload["model"] = model_to_try
                logger.info(f"Trying document analysis with model: {model_to_try}")
                
                for attempt in range(config.OLLAMA_RETRY_ATTEMPTS):
                    try:
                        # GEMMA2:2B optimized timeouts for document analysis
                        timeout_duration = 50 if attempt == 0 else 60  # Document analysis needs more time
                        
                        response = requests.post(
                            f"{ollama_base_url}/api/chat",
                            json=payload,
                            timeout=timeout_duration
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            if 'message' in data and 'content' in data['message']:
                                analysis = data['message']['content']

                                # Extract key medical values if present
                                medical_values = self.extract_medical_values(extracted_text)

                                # Add highlighting to critical medical values in both extracted text and analysis
                                highlighted_extracted_text = self.highlight_critical_values(extracted_text)
                                highlighted_analysis = self.highlight_critical_values(analysis)

                                return {
                                    "success": True,
                                    "analysis": highlighted_analysis,
                                    "original_analysis": analysis,  # Keep original for reference
                                    "medical_values": medical_values,
                                    "highlighted_extracted_text": highlighted_extracted_text,
                                    "model_used": model_to_try,
                                    "timestamp": config.get_timestamp() if hasattr(config, 'get_timestamp') else None,
                                    "attempts": attempt + 1
                                }
                            else:
                                last_error = "Invalid response format from Ollama"
                                break  # Try next model
                        else:
                            last_error = f"HTTP {response.status_code}: {response.text}"
                            if response.status_code == 404:
                                break  # Model not found, try next model
                            
                    except requests.exceptions.Timeout:
                        last_error = f"Document analysis timed out after {timeout_duration} seconds"
                        logger.warning(f"Model {model_to_try} timeout on attempt {attempt + 1}")
                    except requests.exceptions.ConnectionError:
                        last_error = "Cannot connect to Ollama service"
                        logger.warning(f"Connection error with {model_to_try} on attempt {attempt + 1}")
                    except Exception as e:
                        last_error = f"Unexpected error: {str(e)}"
                        logger.error(f"Unexpected error with {model_to_try} on attempt {attempt + 1}: {e}")
                    
                    # Fast retry strategy for speed
                    if attempt < config.OLLAMA_RETRY_ATTEMPTS - 1:
                        time.sleep(0.5 + attempt * 0.3)  # Very fast retry: 0.5s, 0.8s
                
                logger.warning(f"Model {model_to_try} failed for document analysis")
            
            # All attempts failed - provide fast fallback analysis
            logger.warning(f"LLM analysis failed, providing fallback analysis")
            
            # Fast fallback - extract basic medical info from text
            fallback_analysis = self._create_fallback_analysis(extracted_text)
            
            return {
                "success": True,
                "analysis": fallback_analysis,
                "fallback": True,
                "original_error": last_error
            }
            
        except Exception as e:
            logger.error(f"Error analyzing medical document: {e}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "error_type": "analysis_error"
            }
    
    def _create_fallback_analysis(self, text: str) -> str:
        """Create a fast fallback analysis when LLM fails."""
        try:
            # Extract medical values using existing method
            medical_values = self.extract_medical_values(text)
            
            # Basic document type detection
            text_lower = text.lower()
            if any(term in text_lower for term in ['lab', 'blood', 'test', 'result']):
                doc_type = "Lab Report"
            elif any(term in text_lower for term in ['prescription', 'medication', 'mg', 'tablet']):
                doc_type = "Prescription"
            else:
                doc_type = "Medical Document"
            
            # Create simple analysis
            analysis = f"**Document Type:** {doc_type}\n\n"
            analysis += "**Key Findings:**\n"
            
            if medical_values:
                for value in medical_values[:5]:  # Limit to top 5 values
                    analysis += f"• {value['name']}: {value['value']} {value.get('unit', '')}\n"
            else:
                # Extract any numbers that might be medical values
                import re
                numbers = re.findall(r'\b\d+\.?\d*\b', text)
                if numbers:
                    analysis += f"• Document contains numerical values: {', '.join(numbers[:5])}\n"
                else:
                    analysis += "• Document text extracted successfully\n"
            
            analysis += f"\n**Summary:** This {doc_type.lower()} has been processed and key medical values extracted. "
            analysis += "Advanced AI analysis is currently processing - please review the extracted values above.\n\n"
            analysis += "**Next Steps:** \n1. Review the medical values listed above\n2. Compare any lab values to their normal ranges\n3. Discuss these results with your healthcare provider for detailed clinical interpretation"
            
            return analysis
            
        except Exception as e:
            return f"**Document Type:** Medical Document\n\n**Summary:** Document processed successfully. Text extraction completed but detailed analysis is temporarily unavailable. Please consult your healthcare provider for interpretation.\n\n**Extracted Text Length:** {len(text)} characters"
    
    def extract_medical_values(self, text: str) -> List[Dict[str, str]]:
        """Extract key medical values from text using regex patterns."""
        try:
            medical_values = []
            
            # Common lab value patterns
            patterns = [
                # CBC values
                (r'WBC[:\s]+([0-9.]+)', 'WBC'),
                (r'RBC[:\s]+([0-9.]+)', 'RBC'), 
                (r'Hgb|Hemoglobin[:\s]+([0-9.]+)', 'Hemoglobin'),
                (r'Hct|Hematocrit[:\s]+([0-9.]+)', 'Hematocrit'),
                (r'Platelets[:\s]+([0-9.]+)', 'Platelets'),
                
                # Chemistry values
                (r'Glucose[:\s]+([0-9.]+)', 'Glucose'),
                (r'Cholesterol[:\s]+([0-9.]+)', 'Total Cholesterol'),
                (r'HDL[:\s]+([0-9.]+)', 'HDL'),
                (r'LDL[:\s]+([0-9.]+)', 'LDL'),
                (r'Triglycerides[:\s]+([0-9.]+)', 'Triglycerides'),
                
                # Vital signs
                (r'BP|Blood Pressure[:\s]+([0-9]+/[0-9]+)', 'Blood Pressure'),
                (r'HR|Heart Rate[:\s]+([0-9]+)', 'Heart Rate'),
                (r'Temperature[:\s]+([0-9.]+)', 'Temperature'),
                (r'Weight[:\s]+([0-9.]+)', 'Weight'),
            ]
            
            for pattern, name in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    medical_values.append({
                        "name": name,
                        "value": match,
                        "unit": self.get_unit_for_value(name)
                    })
            
            return medical_values
            
        except Exception as e:
            logger.error(f"Error extracting medical values: {e}")
            return []
    
    def get_unit_for_value(self, value_name: str) -> str:
        """Get appropriate unit for medical value."""
        units = {
            "WBC": "K/uL",
            "RBC": "M/uL", 
            "Hemoglobin": "g/dL",
            "Hematocrit": "%",
            "Platelets": "K/uL",
            "Glucose": "mg/dL",
            "Total Cholesterol": "mg/dL",
            "HDL": "mg/dL",
            "LDL": "mg/dL",
            "Triglycerides": "mg/dL",
            "Blood Pressure": "mmHg",
            "Heart Rate": "bpm",
            "Temperature": "°F",
            "Weight": "lbs"
        }
        return units.get(value_name, "")
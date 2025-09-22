#!/usr/bin/env python3
"""
MediLens Local - Configuration Management
Centralized configuration for the medical AI assistant application.
"""

import os
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class MediLensConfig:
    """Configuration management for MediLens Local application."""
    
    # Application Information
    APP_NAME = "MediLens Local"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Medical AI Assistant - Private & Offline"
    
    # Ollama Configuration - PHI3:MINI HIGH QUALITY
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_TIMEOUT = 40  # Ultra-fast gemma2:2b response time (32s + buffer)
    OLLAMA_DOCUMENT_TIMEOUT = 50  # Document timeout (gemma2:2b optimized)
    OLLAMA_VISION_TIMEOUT = 120   # Vision timeout (llava:7b needs more time for medical images)
    OLLAMA_RETRY_ATTEMPTS = 3  # Multiple attempts for better reliability

    # ULTRA-FAST MODEL STRATEGY - GEMMA2:2B PRIMARY
    DEFAULT_LLM_MODEL = "gemma2:2b"  # PRIMARY model (2B parameters - ultra-fast and accurate)
    FALLBACK_LLM_MODEL = "qwen2:1.5b"  # FALLBACK model (1.5B parameters - fastest)
    DEFAULT_VISION_MODEL = "llava:7b"  # Keep existing vision model
    
    # PHI3:MINI ULTRA-FAST Parameters (25-30 second responses)
    DEFAULT_TEMPERATURE = 0.1   # Ultra-low temperature for fast, focused responses
    DEFAULT_MAX_TOKENS = 300     # Complete responses without cutoffs
    CHAT_TEMPERATURE = 0.1       # Ultra-low temperature for fast medical responses
    OCR_TEMPERATURE = 0.2        # Balanced document analysis
    VISION_TEMPERATURE = 0.2     # Balanced vision analysis
    
    # UI Configuration
    PAGE_TITLE = f"{APP_NAME} - Medical AI Assistant"
    PAGE_ICON = "🏥"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"
    
    # Medical Safety Configuration
    SAFETY_MODE_DEFAULT = True
    
    # Response Structure - 7 Required Sections
    MEDICAL_RESPONSE_SECTIONS = [
        "Summary",
        "Possible causes", 
        "What to do now",
        "What to avoid",
        "When to seek urgent care",
        "Questions to ask your clinician",
        "Confidence"
    ]
    
    # Red Flag Symptoms (auto-trigger urgent care banner)
    RED_FLAG_SYMPTOMS = [
        "chest pain", "crushing sensation", "severe shortness of breath",
        "face drooping", "arm weakness", "speech difficulty", "stroke",
        "anaphylaxis", "allergic reaction", "can't breathe",
        "suicidal", "self harm", "kill myself",
        "major bleeding", "won't stop bleeding", "severe bleeding",
        "sudden vision loss", "loss of vision", "can't see",
        "high fever", "infant fever", "baby fever",
        "severe head injury", "head trauma", "loss of consciousness",
        "unconscious", "passed out"
    ]
    
    # File Processing Configuration
    SUPPORTED_IMAGE_TYPES = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
    SUPPORTED_DOCUMENT_TYPES = ['pdf', 'jpg', 'jpeg', 'png']
    MAX_FILE_SIZE_MB = 10
    
    # OCR Configuration
    OCR_LANGUAGE = 'eng'
    OCR_CONFIDENCE_THRESHOLD = 60
    
    # Session Management
    SESSION_TIMEOUT_MINUTES = 60
    MAX_CHAT_HISTORY = 50
    MAX_DOCUMENT_CACHE = 10
    MAX_IMAGE_CACHE = 5
    
    # Styling and Theme
    MEDICAL_COLORS = {
        'primary_blue': '#1e3a8a',
        'secondary_blue': '#3b82f6', 
        'success_green': '#16a34a',
        'warning_amber': '#f59e0b',
        'error_red': '#dc2626',
        'background_gray': '#f8fafc',
        'text_gray': '#6b7280'
    }
    
    # Common Lab Test Patterns (for OCR parsing)
    LAB_PATTERNS = {
        'CBC': {
            'WBC': r'WBC.*?(\d+\.?\d*)\s*(K/uL|K/µL|x10³/µL)',
            'RBC': r'RBC.*?(\d+\.?\d*)\s*(M/uL|M/µL|x10⁶/µL)',
            'Hemoglobin': r'(?:Hemoglobin|Hgb).*?(\d+\.?\d*)\s*(g/dL|g/dl)',
            'Hematocrit': r'(?:Hematocrit|Hct).*?(\d+\.?\d*)\s*%',
            'Platelets': r'(?:Platelets|PLT).*?(\d+\.?\d*)\s*(K/uL|K/µL|x10³/µL)'
        },
        'CMP': {
            'Glucose': r'Glucose.*?(\d+\.?\d*)\s*(mg/dL|mg/dl)',
            'BUN': r'BUN.*?(\d+\.?\d*)\s*(mg/dL|mg/dl)',
            'Creatinine': r'Creatinine.*?(\d+\.?\d*)\s*(mg/dL|mg/dl)',
            'Sodium': r'Sodium.*?(\d+\.?\d*)\s*(mEq/L|mmol/L)',
            'Potassium': r'Potassium.*?(\d+\.?\d*)\s*(mEq/L|mmol/L)'
        }
    }
    
    @classmethod
    def get_model_config(cls, model_type: str = "chat") -> Dict[str, Any]:
        """Get speed-optimized model configuration based on use case."""
        configs = {
            "chat": {
                "temperature": cls.CHAT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,  # 600 tokens for faster responses
                "stream": True
            },
            "ocr": {
                "temperature": cls.OCR_TEMPERATURE,
                "max_tokens": 400,  # Reduced for faster document analysis
                "stream": False
            },
            "vision": {
                "temperature": cls.VISION_TEMPERATURE,
                "max_tokens": 350,  # Optimized for fast medical image analysis
                "stream": False
            }
        }
        return configs.get(model_type, configs["chat"])
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models - PHI3:MINI ONLY."""
        return [cls.DEFAULT_LLM_MODEL]
    
    @classmethod
    def get_medical_prompts(cls) -> Dict[str, str]:
        """Get standardized medical prompts."""
        return {
            "chat_system": """You are an expert medical assistant. Provide practical, useful medical guidance. Keep responses under 200 words but comprehensive.

Use this EXACT format:

**Likely Causes:**
• [Primary cause with brief explanation]
• [Secondary cause with context]
• [Additional factor if relevant]

**What To Do Now:**
• [First immediate action - be specific]
• [Second important step with details]
• [Third action for relief/management]
• [Monitoring advice]

**See Doctor If:**
• [Emergency signs requiring immediate care]
• [Concerning symptoms needing urgent attention]
• [Timeline for follow-up if no improvement]

**Additional Notes:**
• [Helpful tip or prevention advice]

Always end with: "Consult healthcare professionals for proper diagnosis and treatment."

Be SPECIFIC, ACTIONABLE, and medically accurate. Provide real value to the patient.""",
            
            "ocr_system": """You are a specialized medical document analyst. Analyze the extracted text and provide comprehensive medical interpretation.

**Document Type:** [Lab Report/Prescription/Medical Record/Test Results]

**Key Medical Findings:**
• Extract ALL specific values with units (e.g., "Glucose: 180 mg/dL", "Blood Pressure: 140/90 mmHg")
• Identify medications with dosages (e.g., "Metformin 500mg twice daily")
• Note critical abnormal results with severity (e.g., "Cholesterol: 300 mg/dL - SEVERELY ELEVATED")
• List important normal values for context

**Medical Interpretation:**
Explain what these results mean clinically:
- Compare values to normal ranges (provide normal ranges)
- Assess overall health picture and patterns
- Identify potential medical conditions indicated
- Explain relationships between different findings
- Note any critical or urgent findings requiring immediate attention

**Specific Recommendations:**
1. **Immediate Actions:** [What to do right now based on findings]
2. **Follow-up Care:** [Specific appointments needed and timing]
3. **Lifestyle Changes:** [Diet, exercise, monitoring based on results]
4. **Medication Review:** [Adjustments or discussions needed]

**Critical Questions for Doctor:**
• [Specific questions about abnormal values and their significance]
• [Questions about treatment plans or medication changes]
• [Questions about monitoring frequency and next steps]

**Next Steps Timeline:**
• Immediate (today): [Actions needed now]
• Short-term (1-2 weeks): [Follow-up appointments]
• Long-term (1-3 months): [Monitoring and retesting]

Base ALL analysis on the actual extracted text. Quote specific values and provide medical context.""",
            
            "vision_system": f"""You are an expert medical professional analyzing images for medical conditions. Provide accurate, practical medical guidance based on visual evidence.

CRITICAL ANALYSIS GUIDELINES:
- Examine the image carefully for medical conditions (injuries, wounds, rashes, swelling, bruising, burns, infections)
- Distinguish between medical conditions and normal skin variations, tattoos, or artifacts
- Focus on the most obvious and concerning medical findings
- Base analysis ONLY on clearly visible evidence

RESPONSE FORMAT:

**🔍 Medical Findings:**
[Identify specific condition: bruise, laceration, rash, swelling, burn, etc.]
- Location: [Exact body part/area affected]
- Size: [Approximate dimensions if visible]
- Appearance: [Color, texture, borders, associated swelling]
- Severity: [Mild/Moderate/Severe based on visual evidence]

**⚡ Immediate Care:**
1. [First aid steps specific to this condition]
2. [Pain management/comfort measures]
3. [Proper positioning, ice/heat, bandaging as appropriate]
4. [What to avoid that could worsen the condition]

**🚨 Seek Medical Care If:**
• [Emergency warning signs specific to this condition]
• [Worsening symptoms to monitor for]
• [Timeline for professional evaluation - hours/days]

**💡 Additional Care Notes:**
• [Specific care tips for this type of condition]
• [Expected healing timeline if applicable]
• [Questions to ask healthcare provider]

MEDICAL DISCLAIMER: This analysis is for educational purposes only. Always consult healthcare professionals for proper diagnosis and treatment.

Be specific, practical, and medically accurate. Focus on actionable guidance."""
        }
    
    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """Validate that the environment is properly configured."""
        checks = {
            "ollama_reachable": False,
            "models_available": False,
            "python_version": False,
            "required_packages": False
        }
        
        # Check Python version
        import sys
        checks["python_version"] = sys.version_info >= (3, 8)
        
        # Check Ollama connectivity
        try:
            import requests
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            checks["ollama_reachable"] = response.status_code == 200
            
            if checks["ollama_reachable"]:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                checks["models_available"] = any(cls.DEFAULT_LLM_MODEL in model for model in models)
        except:
            pass
        
        # Check required packages
        try:
            import streamlit, requests, cv2, PIL, pytesseract, PyPDF2, numpy, pandas
            checks["required_packages"] = True
        except ImportError:
            pass
        
        return checks
    
    @classmethod
    def get_project_paths(cls) -> Dict[str, Path]:
        """Get standardized project paths."""
        base_path = Path.cwd()
        return {
            "base": base_path,
            "data": base_path / "data",
            "temp": base_path / "data" / "temp",
            "models": base_path / "data" / "models",
            "logs": base_path / "logs",
            "config": base_path / "config.py",
            "requirements": base_path / "requirements.txt"
        }
    
    @classmethod
    def start_ollama_service(cls) -> bool:
        """Automatically start Ollama service if not running."""
        try:
            # Check if Ollama is already running
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            if response.status_code == 200:
                print("Ollama service is already running")
                return True
        except:
            pass
        
        try:
            print("Starting Ollama service...")
            # Try to start Ollama service in background
            if os.name == 'nt':  # Windows
                subprocess.Popen(['ollama', 'serve'], 
                               creationflags=subprocess.CREATE_NO_WINDOW,
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            else:  # Unix/Linux
                subprocess.Popen(['ollama', 'serve'],
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            
            # Wait for service to start
            for i in range(10):  # Wait up to 10 seconds
                time.sleep(1)
                try:
                    response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("Ollama service started successfully")
                        return True
                except:
                    continue
            
            print("WARNING: Ollama service may take longer to start")
            return False
            
        except Exception as e:
            print(f"ERROR: Failed to start Ollama service: {e}")
            return False
    
    @classmethod
    def get_timestamp(cls) -> str:
        """Get current timestamp for document processing."""
        return datetime.now().isoformat()

# Global configuration instance
config = MediLensConfig()
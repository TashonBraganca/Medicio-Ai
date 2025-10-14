#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediLens Local - Configuration Management
Centralized configuration for the medical AI assistant application.
"""

import os
import sys
import platform
import subprocess
import time
import requests
import socket
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime

class MediLensConfig:
    """Configuration management for MediLens Local application."""
    
    # Application Information
    APP_NAME = "MediLens Local"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Medical AI Assistant - Private & Offline"
    
    # Universal Ollama Configuration - Cross-Platform Compatible
    OLLAMA_DEFAULT_PORT = 11434
    OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_DEFAULT_PORT}"

    # Progressive timeout strategy for different systems
    OLLAMA_TIMEOUT = 60  # Increased for macOS M1/M2 compatibility
    OLLAMA_DOCUMENT_TIMEOUT = 80  # Extended for complex document analysis
    OLLAMA_VISION_TIMEOUT = 150   # Extended for vision models on all platforms
    OLLAMA_RETRY_ATTEMPTS = 5  # More retries for better cross-platform reliability
    OLLAMA_CONNECTION_TIMEOUT = 10  # Connection timeout for service detection

    # Alternative ports to try if default fails
    OLLAMA_FALLBACK_PORTS = [11434, 11435, 11436, 8080, 8081]

    # OS-specific timeouts
    OS_TIMEOUT_MULTIPLIERS = {
        'Darwin': 1.5,  # macOS needs more time
        'Windows': 1.0,
        'Linux': 1.2
    }

    # Model Configuration - Medical-Optimized Hierarchy
    # Hierarchical model selection from best medical-specific to fastest general models
    MODEL_HIERARCHY = [
        "meditron:7b",    # 👑 PREMIUM MEDICAL - Trained on medical literature, outperforms GPT-3.5 on medical tasks (~40-60s)
        "llama3.1:8b",    # High-quality general model, comprehensive responses (~60-90s)
        "mistral:7b",     # High-quality alternative general model (~50-75s)
        "gemma2:2b",      # Fast - Good balance of speed and accuracy (~25-35s)
        "qwen2:1.5b",     # Ultra-fast fallback (~20-30s)
    ]

    DEFAULT_LLM_MODEL = "meditron:7b"  # MEDICAL EXPERT MODEL - Trained on medical literature (default)
    FALLBACK_LLM_MODEL = "gemma2:2b"  # Fast fallback if meditron not available
    DEFAULT_VISION_MODEL = "llava:7b"

    # Auto-download models if missing (DISABLED - models persist in Ollama storage)
    AUTO_DOWNLOAD_MODELS = False  # Models already downloaded persist permanently in Ollama
    ESSENTIAL_MODELS = ["meditron:7b", "gemma2:2b", "qwen2:1.5b", "llava:7b"]  # meditron:7b added for medical accuracy

    # Medical Response Parameters - Optimized for CONCISE yet DETAILED responses with medication tables
    DEFAULT_TEMPERATURE = 0.3   # Balanced for natural, personalized responses
    DEFAULT_MAX_TOKENS = 550     # REDUCED by 8% (from 600) - concise but complete medicine table and key instructions
    CHAT_TEMPERATURE = 0.3       # Natural medical responses with personality
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
        """Get comprehensive model configuration based on use case."""
        configs = {
            "chat": {
                "temperature": cls.CHAT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,  # 550 tokens (reduced by 8%) for CONCISE but complete responses
                "stream": True
            },
            "ocr": {
                "temperature": cls.OCR_TEMPERATURE,
                "max_tokens": 600,  # Increased for detailed document analysis
                "stream": False
            },
            "vision": {
                "temperature": cls.VISION_TEMPERATURE,
                "max_tokens": 500,  # Increased for thorough medical image analysis
                "stream": False
            }
        }
        return configs.get(model_type, configs["chat"])
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models."""
        return [cls.DEFAULT_LLM_MODEL]
    
    @classmethod
    def get_medical_prompts(cls) -> Dict[str, str]:
        """Get standardized medical prompts."""
        return {
            "chat_system": """You are an expert medical advisor. Provide CONCISE, USEFUL responses with specific actionable advice. CRITICAL REQUIREMENT: You MUST include a complete medication table.

**🩺 CONDITION & RISK**
• **[Specific Condition/Disease Name]** | Risk: [🟢 LOW / 🟡 MODERATE / 🔴 HIGH - URGENT]
• **Brief explanation:** [1-2 sentences about what this is and why it happens]

**💊 IMMEDIATE ACTIONS** (**step-by-step**):
1. **[Action #1]:** [Specific instructions - be concise but clear]
2. **[Action #2]:** [Specific instructions with key details]
3. **[Action #3]:** [Detailed guidance with specifics]

**💊 RECOMMENDED INDIAN MEDICINES**

MANDATORY: Include complete medication table in EXACT markdown format:

| **Medicine Name** | **Dosage** | **How Many Times/Day** | **For How Many Days** | **Purpose** |
|---------------|--------|-------------------|-------------------|---------|
| [Indian Brand] | [Exact amount] | [Specific times] | [Exact days] | [What it treats] |
| [Indian Brand] | [Exact amount] | [Specific times] | [Exact days] | [What it treats] |
| [Indian Brand] | [Exact amount] | [Specific times] | [Exact days] | [What it treats] |

**EXAMPLE (Always format like this):**
| **Medicine Name** | **Dosage** | **How Many Times/Day** | **For How Many Days** | **Purpose** |
|---------------|--------|-------------------|-------------------|---------|
| **Dolo 650** | 650mg tablet | 3 times (morning, afternoon, night) | 5 days | Reduces fever and body pain |
| **Sinarest** | 1 tablet | 2 times (morning, evening) | 3 days | Relieves cold, congestion, runny nose |
| **ORS Solution** | 1 sachet in 1L water | 3-4 times | Until symptoms improve | Prevents dehydration, replaces electrolytes |

Use specific Indian brands: Dolo 650, Crocin 500, Combiflam, Sinarest, Zincovit, ORS, Cetrizine, Paracetamol, etc.

**🏥 WHEN TO SEE DOCTOR:**
• **[Specific symptom]** that requires doctor visit
• **[Another warning sign]** to watch for
• **Timeline:** [When exactly to visit - e.g., "If no improvement in 48 hours"]

**📋 ADDITIONAL TIPS:**
• **Diet:** [Helpful diet tip]
• **Lifestyle:** [Key lifestyle advice]
• **Monitor:** [What to watch]

Keep response FOCUSED and CONCISE. Be specific with doses, times, and instructions. Make all key points and topics BOLD for easy reading.""",

            "ocr_system": """You are a specialized medical document analyst. Analyze the extracted text and provide comprehensive medical interpretation.

**Document Type:** [Lab Report/Prescription/Medical Record/Test Results]

**Key Medical Findings:**
- Extract ALL specific values with units (e.g., "Glucose: 180 mg/dL", "Blood Pressure: 140/90 mmHg")
- Identify medications with dosages (e.g., "Metformin 500mg twice daily")
- Note critical abnormal results with severity (e.g., "Cholesterol: 300 mg/dL - SEVERELY ELEVATED")
- List important normal values for context

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
- [Specific questions about abnormal values and their significance]
- [Questions about treatment plans or medication changes]
- [Questions about monitoring frequency and next steps]

**Next Steps Timeline:**
- Immediate (today): [Actions needed now]
- Short-term (1-2 weeks): [Follow-up appointments]
- Long-term (1-3 months): [Monitoring and retesting]

Base ALL analysis on the actual extracted text. Quote specific values and provide medical context.""",
            
            "vision_system": """You are an expert medical professional analyzing images for medical conditions. Provide accurate, practical medical guidance based on visual evidence.

CRITICAL ANALYSIS GUIDELINES:
- Examine the image carefully for medical conditions (injuries, wounds, rashes, swelling, bruising, burns, infections)
- Distinguish between medical conditions and normal skin variations, tattoos, or artifacts
- Focus on the most obvious and concerning medical findings
- Base analysis ONLY on clearly visible evidence

RESPONSE FORMAT:

**MEDICAL FINDINGS:**
[Identify specific condition: bruise, laceration, rash, swelling, burn, etc.]
- Location: [Exact body part/area affected]
- Size: [Approximate dimensions if visible]
- Appearance: [Color, texture, borders, associated swelling]
- Severity: [Mild/Moderate/Severe based on visual evidence]

**IMMEDIATE CARE:**
1. [First aid steps specific to this condition]
2. [Pain management/comfort measures]
3. [Proper positioning, ice/heat, bandaging as appropriate]
4. [What to avoid that could worsen the condition]

**SEEK MEDICAL CARE IF:**
- [Emergency warning signs specific to this condition]
- [Worsening symptoms to monitor for]
- [Timeline for professional evaluation - hours/days]

**ADDITIONAL CARE NOTES:**
- [Specific care tips for this type of condition]
- [Expected healing timeline if applicable]
- [Questions to ask healthcare provider]

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
    def detect_ollama_service(cls) -> Tuple[bool, str, str]:
        """Universal Ollama service detection across all platforms."""
        # Try different ports and interfaces
        test_urls = [
            f"http://localhost:{port}" for port in cls.OLLAMA_FALLBACK_PORTS
        ] + [
            f"http://127.0.0.1:{port}" for port in cls.OLLAMA_FALLBACK_PORTS
        ]

        # Add OS-specific URLs
        if platform.system() == 'Darwin':  # macOS
            test_urls.extend([
                f"http://0.0.0.0:{port}" for port in cls.OLLAMA_FALLBACK_PORTS
            ])

        for url in test_urls:
            try:
                response = requests.get(f"{url}/api/tags", timeout=3)
                if response.status_code == 200:
                    return True, url, f"Found Ollama at {url}"
            except:
                continue

        return False, "", "Ollama service not detected on any standard port"

    @classmethod
    def start_ollama_service(cls) -> Tuple[bool, str]:
        """Universal Ollama service startup for all platforms."""
        # First check if already running
        is_running, url, message = cls.detect_ollama_service()
        if is_running:
            cls.OLLAMA_BASE_URL = url  # Update to working URL
            return True, f"Ollama already running at {url}"

        try:
            print("Starting Ollama service...")
            system_os = platform.system()

            # OS-specific service startup
            if system_os == 'Windows':
                # Windows startup
                subprocess.Popen(['ollama', 'serve'],
                               creationflags=subprocess.CREATE_NO_WINDOW,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            elif system_os == 'Darwin':
                # macOS startup with proper environment
                env = os.environ.copy()
                env['OLLAMA_HOST'] = '0.0.0.0:11434'  # Bind to all interfaces on macOS
                subprocess.Popen(['ollama', 'serve'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               env=env)
            else:
                # Linux/Unix startup
                subprocess.Popen(['ollama', 'serve'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)

            # Progressive wait with OS-specific timeouts
            wait_time = 15 if system_os == 'Darwin' else 10

            for i in range(wait_time):
                time.sleep(1)
                is_running, url, message = cls.detect_ollama_service()
                if is_running:
                    cls.OLLAMA_BASE_URL = url  # Update to working URL
                    return True, f"Ollama started successfully at {url}"

            return False, f"Ollama service startup timeout after {wait_time}s"

        except FileNotFoundError:
            return False, "Ollama command not found. Please install Ollama first."
        except Exception as e:
            return False, f"Failed to start Ollama: {str(e)}"

    @classmethod
    def ensure_models_available(cls) -> Tuple[bool, List[str], List[str]]:
        """Ensure essential models are available, download if missing."""
        try:
            # Get currently installed models
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags",
                                  timeout=cls.OLLAMA_CONNECTION_TIMEOUT)
            if response.status_code != 200:
                return False, [], ["Cannot connect to Ollama service"]

            data = response.json()
            installed_models = [model['name'] for model in data.get('models', [])]

            available_models = []
            missing_models = []

            # Check model hierarchy
            for model in cls.MODEL_HIERARCHY:
                if any(model in installed for installed in installed_models):
                    available_models.append(model)
                else:
                    missing_models.append(model)

            # Auto-download essential models if enabled
            if cls.AUTO_DOWNLOAD_MODELS and missing_models:
                print("Auto-downloading essential models...")
                for model in cls.ESSENTIAL_MODELS:
                    if model in missing_models:
                        success = cls._download_model(model)
                        if success:
                            available_models.append(model)
                            missing_models.remove(model)

            return True, available_models, missing_models

        except Exception as e:
            return False, [], [f"Error checking models: {str(e)}"]

    @classmethod
    def _download_model(cls, model_name: str) -> bool:
        """Download a model using Ollama."""
        try:
            print(f"Downloading {model_name}...")

            # Use requests to trigger download
            payload = {"name": model_name}
            response = requests.post(f"{cls.OLLAMA_BASE_URL}/api/pull",
                                   json=payload,
                                   timeout=300,  # 5 minute timeout for downloads
                                   stream=True)

            if response.status_code == 200:
                print(f"Successfully downloaded {model_name}")
                return True
            else:
                print(f"Failed to download {model_name}: {response.status_code}")
                return False

        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            return False

    @classmethod
    def get_best_available_model(cls) -> Optional[str]:
        """Get the best available model from the hierarchy."""
        try:
            success, available_models, _ = cls.ensure_models_available()
            if success and available_models:
                # Return first available model from hierarchy
                for model in cls.MODEL_HIERARCHY:
                    if model in available_models:
                        return model
            return None
        except:
            return None

    @classmethod
    def get_os_optimized_timeout(cls, base_timeout: int) -> int:
        """Get OS-optimized timeout value."""
        system_os = platform.system()
        multiplier = cls.OS_TIMEOUT_MULTIPLIERS.get(system_os, 1.0)
        return int(base_timeout * multiplier)
    
    @classmethod
    def get_timestamp(cls) -> str:
        """Get current timestamp for document processing."""
        return datetime.now().isoformat()

# Global configuration instance
config = MediLensConfig()
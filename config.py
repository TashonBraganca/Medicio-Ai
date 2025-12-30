"""MediLens Local - Configuration Management"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import platform
import subprocess
import time
import os
import requests


DEFAULT_MODEL = "gemma2:2b"
OLLAMA_URL = "http://localhost:11434"


def get_medical_prompts() -> Dict[str, str]:
    """
    Backward-compatible helper for older imports: from config import get_medical_prompts.
    Returns the standardized prompts from MediLensConfig.
    """
    return MediLensConfig.MEDICAL_PROMPTS


class MediLensConfig:
    """Configuration management for MediLens Local application."""

    # Application Information
    APP_NAME = "MediLens Local"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Medical AI Assistant - Private & Offline"

    # Universal Ollama Configuration
    OLLAMA_DEFAULT_PORT = 11434
    OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_DEFAULT_PORT}"
    OLLAMA_TIMEOUT = 120
    OLLAMA_RETRY_ATTEMPTS = 3

    # Model Configuration
    DEFAULT_LLM_MODEL = "gemma2:2b"
    DEFAULT_VISION_MODEL = "llava:7b"

    # Response Parameters
    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 520

    # Medical Analysis Prompts (clean, structured)
    MEDICAL_PROMPTS: Dict[str, str] = {
        "chat_system": (
            "You are an expert medical assistant. Provide concise, accurate, and safe guidance "
            "based strictly on established medical consensus. Never provide harmful, experimental, "
            "or non-evidence-based recommendations."
        ),
        "vision_system": (
            "You are an expert in medical image analysis. Base all conclusions only on what is clearly visible "
            "in the image and established medical knowledge. Clearly state uncertainty when needed."
        ),
        "ocr_system": (
            "You are an expert medical laboratory analyst and clinician. Analyze the extracted report text and "
            "respond in this exact structured format:\n\n"
            "[DOCUMENT TYPE]\n"
            "(Identify the specific type of medical document)\n\n"
            "[KEY MEDICAL FINDINGS]\n"
            "- Parameter: [Value] (Normal Range: [Range]) - [Status]\n"
            "(List ALL relevant parameters with complete information)\n\n"
            "[MEDICAL INTERPRETATION]\n"
            "1. [Primary interpretation based on findings]\n"
            "2. [Analysis of abnormal values]\n"
            "3. [Health implications]\n\n"
            "[RECOMMENDED ACTIONS]\n"
            "1. [Immediate steps if needed]\n"
            "2. [Follow-up requirements]\n"
            "3. [Lifestyle modifications]\n"
            "4. [Medication adjustments]\n\n"
            "[QUESTIONS FOR HEALTHCARE PROVIDER]\n"
            "1. [About abnormal findings]\n"
            "2. [About treatment plan]\n"
            "3. [About follow-up]\n\n"
            "[IMPORTANT NOTES]\n"
            "- [Critical values requiring attention]\n"
            "- [Significant trends]\n"
            "- [Monitoring requirements]\n"
            "- [Drug interactions]\n\n"
            "[RECOMMENDED MEDICATIONS]\n"
            "| Medication Name | Dosage | Frequency | Duration | Purpose |\n"
            "|----------------|--------|-----------|----------|---------|\n"
            "| [Med 1]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n"
            "| [Med 2]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n"
            "| [Med 3]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n\n"
            "Rules:\n"
            "1. Include all key values with units and normal ranges where possible.\n"
            "2. Highlight abnormal and critical values clearly.\n"
            "3. Provide clear, actionable, patient-safe next steps.\n"
            "4. Do NOT invent values that are not in the document.\n"
            "5. Always stay within your role as an AI assistant, not a prescribing doctor."
        ),
    }

    @classmethod
    def get_medical_prompts(cls) -> Dict[str, str]:
        """Get standardized medical prompts."""
        return cls.MEDICAL_PROMPTS

    # Universal Ollama Configuration - Cross-Platform Compatible
    OLLAMA_DEFAULT_PORT = 11434
    OLLAMA_BASE_URL = f"http://localhost:{OLLAMA_DEFAULT_PORT}"

    # Optimized timeout strategy for model operations
    OLLAMA_TIMEOUT = 120
    OLLAMA_DOCUMENT_TIMEOUT = 150
    OLLAMA_VISION_TIMEOUT = 180
    OLLAMA_RETRY_ATTEMPTS = 3
    OLLAMA_CONNECTION_TIMEOUT = 10

    # Alternative ports to try if default fails
    OLLAMA_FALLBACK_PORTS: List[int] = [11434, 11435, 11436, 8080, 8081]

    # OS-specific timeouts
    OS_TIMEOUT_MULTIPLIERS = {
        "Darwin": 1.5,   # macOS
        "Windows": 1.0,
        "Linux": 1.2,
    }

    # Model Configuration - Balanced quality and performance for laptops
    MODEL_HIERARCHY: List[str] = [
        "gemma2:2b",   # Fast and high quality
        "gemma2:9b",   # Higher quality, heavier
        "qwen2:1.5b",  # Very fast fallback
        "meditron:7b", # Medical specialist
        "mistral:7b",  # General strong model
    ]

    DEFAULT_LLM_MODEL = "gemma2:2b"
    FALLBACK_LLM_MODEL = "qwen2:1.5b"
    FALLBACK_FAST_MODEL = "qwen2:1.5b"
    DEFAULT_VISION_MODEL = "llava:7b"

    # Auto-download models if missing (disabled by default)
    AUTO_DOWNLOAD_MODELS = False
    ESSENTIAL_MODELS: List[str] = ["gemma2:2b", "qwen2:1.5b", "llava:7b"]

    # Medical Response Parameters
    DEFAULT_TEMPERATURE = 0.2
    DEFAULT_MAX_TOKENS = 2000
    CHAT_TEMPERATURE = 0.2
    OCR_TEMPERATURE = 0.2
    VISION_TEMPERATURE = 0.2

    # Optimized resource settings
    NUM_CTX = 2048
    REPEAT_PENALTY = 1.15
    TOP_K = 40
    TOP_P = 0.9
    NUM_THREAD = 4

    # UI Configuration
    PAGE_TITLE = f"{APP_NAME} - Medical AI Assistant"
    PAGE_ICON = "🏥"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"

    # Medical Safety Configuration
    SAFETY_MODE_DEFAULT = True

    # Response Structure - 7 Required Sections
    MEDICAL_RESPONSE_SECTIONS: List[str] = [
        "Summary",
        "Possible causes",
        "What to do now",
        "What to avoid",
        "When to seek urgent care",
        "Questions to ask your clinician",
        "Confidence",
    ]

    # Red Flag Symptoms (auto-trigger urgent care banner)
    RED_FLAG_SYMPTOMS: List[str] = [
        "chest pain", "crushing sensation", "severe shortness of breath",
        "face drooping", "arm weakness", "speech difficulty", "stroke",
        "anaphylaxis", "allergic reaction", "can't breathe",
        "suicidal", "self harm", "kill myself",
        "major bleeding", "won't stop bleeding", "severe bleeding",
        "sudden vision loss", "loss of vision", "can't see",
        "high fever", "infant fever", "baby fever",
        "severe head injury", "head trauma", "loss of consciousness",
        "unconscious", "passed out",
    ]

    # File Processing Configuration
    SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "gif", "bmp"]
    SUPPORTED_DOCUMENT_TYPES = ["pdf", "jpg", "jpeg", "png"]
    MAX_FILE_SIZE_MB = 10

    # OCR Configuration
    OCR_LANGUAGE = "eng"
    OCR_CONFIDENCE_THRESHOLD = 60

    # Session Management
    SESSION_TIMEOUT_MINUTES = 60
    MAX_CHAT_HISTORY = 50
    MAX_DOCUMENT_CACHE = 10
    MAX_IMAGE_CACHE = 5

    # Styling and Theme
    MEDICAL_COLORS = {
        "primary_blue": "#1e3a8a",
        "secondary_blue": "#3b82f6",
        "success_green": "#16a34a",
        "warning_amber": "#f59e0b",
        "error_red": "#dc2626",
        "background_gray": "#f8fafc",
        "text_gray": "#6b7280",
    }

    # Common Lab Test Patterns (for OCR parsing)
    LAB_PATTERNS: Dict[str, Dict[str, str]] = {
        "CBC": {
            "WBC": r"WBC.*?(\d+\.?\d*)\s*(K/uL|K/µL|x10³/µL)",
            "RBC": r"RBC.*?(\d+\.?\d*)\s*(M/uL|M/µL|x10⁶/µL)",
            "Hemoglobin": r"(?:Hemoglobin|Hgb).*?(\d+\.?\d*)\s*(g/dL|g/dl)",
            "Hematocrit": r"(?:Hematocrit|Hct).*?(\d+\.?\d*)\s*%",
            "Platelets": r"(?:Platelets|PLT).*?(\d+\.?\d*)\s*(K/uL|K/µL|x10³/µL)",
        },
        "CMP": {
            "Glucose": r"Glucose.*?(\d+\.?\d*)\s*(mg/dL|mg/dl)",
            "BUN": r"BUN.*?(\d+\.?\d*)\s*(mg/dL|mg/dl)",
            "Creatinine": r"Creatinine.*?(\d+\.?\d*)\s*(mg/dL|mg/dl)",
            "Sodium": r"Sodium.*?(\d+\.?\d*)\s*(mEq/L|mmol/L)",
            "Potassium": r"Potassium.*?(\d+\.?\d*)\s*(mEq/L|mmol/L)",
        },
    }

    @classmethod
    def get_model_config(cls, model_type: str = "chat") -> Dict[str, Any]:
        """Get comprehensive model configuration based on use case."""
        configs: Dict[str, Dict[str, Any]] = {
            "chat": {
                "temperature": cls.CHAT_TEMPERATURE,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "stream": False,
            },
            "ocr": {
                "temperature": cls.OCR_TEMPERATURE,
                "max_tokens": 600,
                "stream": False,
            },
            "vision": {
                "temperature": cls.VISION_TEMPERATURE,
                "max_tokens": 500,
                "stream": False,
            },
        }
        return configs.get(model_type, configs["chat"])

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models (simple implementation)."""
        return [cls.DEFAULT_LLM_MODEL]

    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """Validate that the environment is properly configured."""
        checks = {
            "ollama_reachable": False,
            "models_available": False,
            "python_version": False,
            "required_packages": False,
        }

        # Check Python version
        import sys

        checks["python_version"] = sys.version_info >= (3, 8)

        # Check Ollama connectivity
        try:
            response = requests.get(f"{cls.OLLAMA_BASE_URL}/api/tags", timeout=5)
            checks["ollama_reachable"] = response.status_code == 200

            if checks["ollama_reachable"]:
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                checks["models_available"] = any(
                    cls.DEFAULT_LLM_MODEL in model for model in models
                )
        except Exception:
            pass

        # Check required packages
        try:
            import streamlit  # noqa: F401
            import cv2        # noqa: F401
            import PIL        # noqa: F401
            import pytesseract  # noqa: F401
            import PyPDF2     # noqa: F401
            import numpy      # noqa: F401
            import pandas     # noqa: F401

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
            "requirements": base_path / "requirements.txt",
        }

    @classmethod
    def detect_ollama_service(cls) -> "tuple[bool, str, str]":
        """Detect Ollama service across common ports."""
        test_urls: List[str] = (
            [f"http://localhost:{port}" for port in cls.OLLAMA_FALLBACK_PORTS]
            + [f"http://127.0.0.1:{port}" for port in cls.OLLAMA_FALLBACK_PORTS]
        )

        if platform.system() == "Darwin":
            test_urls += [f"http://0.0.0.0:{port}" for port in cls.OLLAMA_FALLBACK_PORTS]

        for url in test_urls:
            try:
                response = requests.get(f"{url}/api/tags", timeout=3)
                if response.status_code == 200:
                    return True, url, f"Found Ollama at {url}"
            except Exception:
                continue

        return False, "", "Ollama service not detected on any standard port"

    @classmethod
    def start_ollama_service(cls) -> "tuple[bool, str]":
        """Attempt to start Ollama service if not running."""
        is_running, url, _ = cls.detect_ollama_service()
        if is_running:
            cls.OLLAMA_BASE_URL = url
            return True, f"Ollama already running at {url}"

        try:
            system_os = platform.system()

            if system_os == "Windows":
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system_os == "Darwin":
                env = os.environ.copy()
                env["OLLAMA_HOST"] = "0.0.0.0:11434"
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                )
            else:
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            wait_time = 15 if system_os == "Darwin" else 10
            for _ in range(wait_time):
                time.sleep(1)
                is_running, url, _ = cls.detect_ollama_service()
                if is_running:
                    cls.OLLAMA_BASE_URL = url
                    return True, f"Ollama started successfully at {url}"

            return False, f"Ollama service startup timeout after {wait_time}s"

        except FileNotFoundError:
            return False, "Ollama command not found. Please install Ollama."
        except Exception as e:
            return False, f"Failed to start Ollama: {str(e)}"

    @classmethod
    def ensure_models_available(cls) -> "tuple[bool, List[str], List[str]]":
        """Ensure essential models are available, optionally download if missing."""
        try:
            response = requests.get(
                f"{cls.OLLAMA_BASE_URL}/api/tags",
                timeout=cls.OLLAMA_CONNECTION_TIMEOUT,
            )
            if response.status_code != 200:
                return False, [], ["Cannot connect to Ollama service"]

            data = response.json()
            installed_models = [m["name"] for m in data.get("models", [])]

            available_models: List[str] = []
            missing_models: List[str] = []

            for model in cls.MODEL_HIERARCHY:
                if any(model in installed for installed in installed_models):
                    available_models.append(model)
                else:
                    missing_models.append(model)

            if cls.AUTO_DOWNLOAD_MODELS and missing_models:
                for model in cls.ESSENTIAL_MODELS:
                    if model in missing_models and cls._download_model(model):
                        available_models.append(model)
                        missing_models.remove(model)

            return True, available_models, missing_models

        except Exception as e:
            return False, [], [f"Error checking models: {str(e)}"]

    @classmethod
    def _download_model(cls, model_name: str) -> bool:
        """Download a model using Ollama."""
        try:
            payload = {"name": model_name}
            response = requests.post(
                f"{cls.OLLAMA_BASE_URL}/api/pull",
                json=payload,
                timeout=300,
                stream=True,
            )
            return response.status_code == 200
        except Exception:
            return False

    @classmethod
    def get_best_available_model(cls) -> Optional[str]:
        """Return the best available model from the hierarchy."""
        try:
            success, available_models, _ = cls.ensure_models_available()
            if success and available_models:
                for model in cls.MODEL_HIERARCHY:
                    if model in available_models:
                        return model
            return None
        except Exception:
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


# Global configuration instance (used by the rest of the app)
config = MediLensConfig()
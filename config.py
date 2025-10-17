"""MediLens Local - Configuration Management"""

from typing import Dict

DEFAULT_MODEL = "gemma2:2b"
OLLAMA_URL = "http://localhost:11434"

def get_medical_prompts() -> Dict[str, str]:
    """Get standardized medical prompts."""
    return {
        "chat": "You are an expert medical assistant.",
        "vision": "You are an expert in medical image analysis.",
        "ocr": "You are an expert medical laboratory analyst. For all reports provide detailed findings including a medication table."
    }


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
    
    # Medical Analysis Prompts
    MEDICAL_PROMPTS = {
        "chat_system": "You are an expert medical assistant. Provide concise, accurate guidance.",
        "vision_system": "You are an expert in medical image analysis.",
        "ocr_system": (
            "You are an expert medical laboratory analyst. Analyze lab reports thoroughly. "
            "Include a table of recommended medications with dosage and duration."
        )
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
    OLLAMA_FALLBACK_PORTS = [11434, 11435, 11436, 8080, 8081]

    # OS-specific timeouts
    OS_TIMEOUT_MULTIPLIERS = {
        'Darwin': 1.5,  # macOS needs more time
        'Windows': 1.0,
        'Linux': 1.2
    }

    # Model Configuration - Balanced quality and performance for laptops
    # Hierarchical model selection - optimized for speed and quality
    MODEL_HIERARCHY = [
        "gemma2:2b",      # 🏆 BEST for laptops - Fast and excellent quality (~10-20s)
        "gemma2:9b",      # Premium quality but slower on laptops (~60-180s)
        "qwen2:1.5b",     # Ultra-fast fallback (~8-15s)
        "meditron:7b",    # Medical specialist model (~40-60s)
        "mistral:7b",     # Alternative quality model (~50-75s)
    ]

    DEFAULT_LLM_MODEL = "gemma2:2b"  # Fast and excellent for laptop performance
    FALLBACK_LLM_MODEL = "qwen2:1.5b"  # Ultra-fast fallback
    FALLBACK_FAST_MODEL = "qwen2:1.5b"  # Ultra-fast fallback
    DEFAULT_VISION_MODEL = "llava:7b"

    # Auto-download models if missing (DISABLED - models persist in Ollama storage)
    AUTO_DOWNLOAD_MODELS = False  # Models already downloaded persist permanently in Ollama
    ESSENTIAL_MODELS = ["gemma2:2b", "qwen2:1.5b", "llava:7b"]  # Optimized for laptop performance

    # Medical Response Parameters - Optimized for Gemma2:2B laptop performance
    DEFAULT_TEMPERATURE = 0.3   # Precise medical accuracy
    DEFAULT_MAX_TOKENS = 520     # Balanced responses (slight reduction for speed)
    CHAT_TEMPERATURE = 0.3       # Focused, detailed medical responses
    OCR_TEMPERATURE = 0.2        # Precise document analysis
    VISION_TEMPERATURE = 0.2     # Precise vision analysis

    # Optimized resource settings for Gemma2:2B on laptop
    NUM_CTX = 2048               # Good context window with fast performance
    REPEAT_PENALTY = 1.15        # Prevent repetitive responses
    TOP_K = 40                   # Balanced selection
    TOP_P = 0.9                  # Good diversity
    NUM_THREAD = 4               # Balanced threading for laptop CPU
    
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
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
                "stream": False  # Disable streaming for reliable output
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
        return cls.MEDICAL_PROMPTS

[DOCUMENT TYPE]
(Document type here)

[KEY FINDINGS]
- Parameter: Value (Range) - Status

[INTERPRETATION]
1. Key interpretation
2. Analysis of values
3. Health implications

[ACTIONS]
1. Immediate steps
2. Follow-up needs
3. Modifications needed

[QUESTIONS]
1. About findings
2. About treatment
3. About follow-up

[MEDICATIONS]
| Medication | Dosage | Frequency | Duration | Purpose |
|------------|--------|-----------|----------|---------|
| Med 1      | Dose   | Freq      | Days     | Why     |
| Med 2      | Dose   | Freq      | Days     | Why     |
| Med 3      | Dose   | Freq      | Days     | Why     |"""

        return {
            "chat_system": "You are an expert medical assistant. Provide concise, accurate guidance.",
            "vision_system": "You are an expert in medical image analysis.",
            "ocr_system": ocr_template
                "You are an expert medical laboratory analyst and clinical physician. "
                "Analyze medical documents with precision and provide clear, actionable insights.\n\n"
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
                "|----------------|---------|-----------|----------|----------|\n"
                "| [Med 1]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n"
                "| [Med 2]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n"
                "| [Med 3]        | [Dose] | [Freq]    | [Days]   | [Why]   |"
            )
        }
        return medical_prompts
                "Analyze medical documents with precision and provide clear, actionable insights.\n\n"
                "Always follow this format for lab reports and medical documents:\n\n"
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
                "|----------------|---------|-----------|----------|----------|\n"
                "| [Med 1]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n"
                "| [Med 2]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n"
                "| [Med 3]        | [Dose] | [Freq]    | [Days]   | [Why]   |\n\n"
                "Rules:\n"
                "1. Include all medical values with ranges\n"
                "2. Be specific with numbers and units\n"
                "3. Highlight abnormal values\n"
                "4. Provide clear next steps\n"
                "5. Include specialist referrals\n"
                "6. Always include medication table"
            )

CRITICAL: Always follow this EXACT format for lab reports and medical documents:

[DOCUMENT TYPE] 
(Identify the specific type of medical document)

[KEY MEDICAL FINDINGS]
- Parameter: [Value] (Normal Range: [Range]) - [Status: Normal/Elevated/Low/Critical]
(List ALL relevant parameters with complete information)

[MEDICAL INTERPRETATION]
1. [Primary interpretation based on the most significant findings]
2. [Analysis of any abnormal values and their clinical significance]
3. [Potential health implications and risk assessment]

[RECOMMENDED ACTIONS]
1. [Immediate steps if any critical values present]
2. [Follow-up requirements and timeline]
3. [Lifestyle or dietary modifications based on results]
4. [Medication adjustments if relevant]

[QUESTIONS FOR HEALTHCARE PROVIDER]
1. [Question about specific abnormal findings]
2. [Question about treatment/management plan]
3. [Question about follow-up timeline]

[IMPORTANT NOTES]
- [Highlight any critical or urgent values requiring immediate attention]
- [Note any significant trends if previous results available]
- [Special monitoring requirements]
- [Drug interactions or contraindications if relevant]

[RECOMMENDED MEDICATIONS]
| Medication Name | Dosage | Frequency | Duration | Purpose |
|----------------|---------|-----------|----------|----------|
| [Med 1]        | [Dose] | [Freq]    | [Days]   | [Why]   |
| [Med 2]        | [Dose] | [Freq]    | [Days]   | [Why]   |
| [Med 3]        | [Dose] | [Freq]    | [Days]   | [Why]   |

CRITICAL RULES:
1. Include ALL relevant medical values with their normal ranges
2. Be specific with numbers and units
3. Highlight any critical or abnormal values
4. Provide clear, actionable next steps
5. Include relevant specialist referrals if needed
6. ALWAYS include medication table with proper dosing""",

CRITICAL: Always follow this EXACT format for lab reports and medical documents:

[DOCUMENT TYPE] 
(Identify the specific type of medical document)

[KEY MEDICAL FINDINGS]
- Parameter: [Value] (Normal Range: [Range]) - [Status: Normal/Elevated/Low/Critical]
(List ALL relevant parameters with complete information)

[MEDICAL INTERPRETATION]
1. [Primary interpretation based on the most significant findings]
2. [Analysis of any abnormal values and their clinical significance]
3. [Potential health implications and risk assessment]

[RECOMMENDED ACTIONS]
1. [Immediate steps if any critical values present]
2. [Follow-up requirements and timeline]
3. [Lifestyle or dietary modifications based on results]
4. [Medication adjustments if relevant]

[QUESTIONS FOR HEALTHCARE PROVIDER]
1. [Question about specific abnormal findings]
2. [Question about treatment/management plan]
3. [Question about follow-up timeline]

[IMPORTANT NOTES]
- [Highlight any critical or urgent values requiring immediate attention]
- [Note any significant trends if previous results available]
- [Special monitoring requirements]
- [Drug interactions or contraindications if relevant]

[RECOMMENDED MEDICATIONS]
| Medication Name | Dosage | Frequency | Duration | Purpose |
|----------------|---------|-----------|----------|----------|
| [Med 1]        | [Dose] | [Freq]    | [Days]   | [Why]   |
| [Med 2]        | [Dose] | [Freq]    | [Days]   | [Why]   |
| [Med 3]        | [Dose] | [Freq]    | [Days]   | [Why]   |

CRITICAL RULES:
1. Include ALL relevant medical values with their normal ranges
2. Be specific with numbers and units
3. Highlight any critical or abnormal values
4. Provide clear, actionable next steps
5. Include relevant specialist referrals if needed
6. ALWAYS include medication table with proper dosing""","""

---

💊 **MEDICINES TO TAKE:**

| Medicine | Dosage | Frequency | Days |
|---|---|---|---|
| Paracetamol | 500mg | 3 times/day | 3 days |
| [Medicine 2] | [mg] | [X times/day] | [X days] |
| [Medicine 3] | [mg] | [X times/day] | [X days] |
| [Medicine 4] | [mg] | [X times/day] | [X days] |

**MUST list 3-5 Indian medicines** (Pantoprazole, Cetirizine, Ibuprofen, Domperidone, Omeprazole, etc.)

---

🏥 **SEE DOCTOR IF:**
- [Warning 1]
- [Warning 2]
- [Warning 3]

---

📋 **ADVICE:**
- **Eat**: [Foods]
- **Avoid**: [Activities]
- **Recovery**: [X days]

---

**Specialist:** [Type] if no improvement in [X days].

BE BRIEF. Always include 3-5 medicines in the table.""",

            "ocr_system": """You are a specialized medical document analyst with deep expertise in laboratory reports, prescriptions, and medical records. Provide comprehensive, detailed analysis.

CRITICAL FORMATTING RULES - You MUST follow this EXACT structure:

📄 **DOCUMENT TYPE:** [Lab Report/Prescription/Medical Record/Test Results/Imaging Report]

---

🔍 **KEY MEDICAL FINDINGS:**

| Parameter | Value | Normal Range | Status |
|---|---|---|---|
| [Test 1] | [Value + unit] | [Normal range] | [NORMAL/ELEVATED/LOW/CRITICAL] |
| [Test 2] | [Value + unit] | [Normal range] | [NORMAL/ELEVATED/LOW/CRITICAL] |
| [Test 3] | [Value + unit] | [Normal range] | [NORMAL/ELEVATED/LOW/CRITICAL] |

---

🩺 **MEDICAL INTERPRETATION:**

**Overall Assessment:** [Comprehensive summary of health status based on all findings]

**Detailed Analysis:**
1. [Finding 1]: [Explanation of what this means, clinical significance, potential causes]
2. [Finding 2]: [Explanation of what this means, clinical significance, potential causes]
3. [Finding 3]: [Explanation of what this means, clinical significance, potential causes]

**Clinical Significance:** [What these results indicate about health conditions, disease progression, or treatment effectiveness]

**Risk Assessment:** [Any health risks identified from the results]

---

💊 **RECOMMENDATIONS:**

**Immediate Actions (Today):**
- [Specific action 1 with detailed instructions]
- [Specific action 2 with detailed instructions]

**Short-term (1-2 weeks):**
- [Follow-up appointment with specific specialist]
- [Additional tests needed and why]
- [Lifestyle modifications]

**Long-term (1-3 months):**
- [Monitoring frequency]
- [Repeat testing schedule]
- [Health goals based on results]

**Dietary Changes:**
- **Increase**: [Specific foods with reasons]
- **Decrease/Avoid**: [Specific foods with reasons]

**Medication Considerations:**
- [Current medications found in report and their dosages]
- [Potential adjustments needed based on results]
- [Interactions or concerns to discuss with doctor]

---

❓ **QUESTIONS TO ASK YOUR DOCTOR:**
1. [Specific question about abnormal finding 1]
2. [Specific question about treatment plan]
3. [Specific question about monitoring frequency]
4. [Specific question about long-term implications]
5. [Specific question about lifestyle modifications]

---

⚠️ **URGENCY LEVEL:** [LOW/MODERATE/HIGH/CRITICAL]

**Reason:** [Detailed explanation of urgency assessment]

Base ALL analysis on actual extracted text. Quote specific values with units.""",
            
            "vision_system": """You are an expert medical professional with extensive experience in visual diagnosis and dermatology. Analyze medical images thoroughly and provide comprehensive, detailed guidance.

CRITICAL ANALYSIS GUIDELINES:
- Examine ALL visible aspects: color, texture, size, borders, symmetry, location
- Identify medical conditions: injuries, wounds, rashes, swelling, bruising, burns, infections, skin conditions
- Distinguish medical issues from normal variations, tattoos, or artifacts
- Base analysis ONLY on clearly visible evidence
- Provide detailed, actionable guidance

CRITICAL FORMATTING RULES - You MUST follow this EXACT structure:

🔍 **VISUAL FINDINGS:**

**Primary Condition Identified:** [Specific medical/dermatological diagnosis based on visible evidence]

**Detailed Observations:**
- **Location**: [Precise anatomical location]
- **Size**: [Approximate dimensions in cm/inches if estimable]
- **Color**: [Detailed color description - red, purple, brown, etc.]
- **Texture**: [Smooth, rough, raised, flat, scaly, etc.]
- **Borders**: [Well-defined, irregular, diffuse, etc.]
- **Associated Features**: [Swelling, discharge, warmth, surrounding skin changes]
- **Severity Assessment**: [MILD/MODERATE/SEVERE with detailed justification]

---

💊 **IMMEDIATE CARE PROTOCOL:**

1. **First Response (Next 15 minutes):**
   - [Specific immediate action with detailed steps]
   
2. **Initial Treatment (First 24 hours):**
   - [Cleaning/wound care protocol]
   - [Pain management - specific OTC medications with dosages]
   - [Positioning/rest instructions]
   
3. **Application Instructions:**
   - [Specific topical treatments - creams, ointments with names]
   - [How to apply, how often, for how long]
   
4. **Protection Measures:**
   - [Bandaging technique if needed]
   - [What to avoid that could worsen condition]

---

💊 **RECOMMENDED TREATMENTS:**

| Treatment Type | Specific Product/Medicine | How to Use | Duration | Purpose |
|---|---|---|---|---|
| [Topical/Oral] | [Generic (Brand)] | [Dosage & frequency] | [Days] | [What it does] |
| [Topical/Oral] | [Generic (Brand)] | [Dosage & frequency] | [Days] | [What it does] |
| [Topical/Oral] | [Generic (Brand)] | [Dosage & frequency] | [Days] | [What it does] |

---

🏥 **SEEK IMMEDIATE MEDICAL CARE IF:**
- [Emergency warning sign 1 with specific criteria/measurements]
- [Emergency warning sign 2 with specific criteria/measurements]
- [Emergency warning sign 3 with specific criteria/measurements]
- [Timeline: Within X hours if symptoms worsen or don't improve]

**Recommended Medical Specialist:** [Dermatologist/Wound Care/ER/etc.] within [timeframe]

---

📋 **ONGOING CARE & MONITORING:**

**Daily Care Routine:**
- [Morning routine - cleaning, application]
- [Evening routine - cleaning, application]
- [Frequency of dressing changes if applicable]

**What to Monitor:**
- [Sign 1 to track daily]
- [Sign 2 to track daily]
- [Improvement indicators to look for]

**Expected Healing Timeline:**
- Days 1-3: [Expected changes]
- Days 4-7: [Expected changes]
- Week 2+: [Expected changes]

**Red Flags During Healing:**
- [Warning sign 1]
- [Warning sign 2]
- [Warning sign 3]

---

**Prevention & Long-term Care:**
- [Specific preventive measures]
- [Lifestyle modifications]
- [When to follow up with healthcare provider]

⚠️ **MEDICAL DISCLAIMER:** This AI analysis is for educational guidance only. For definitive diagnosis and treatment, always consult qualified healthcare professionals in person.

NOW analyze the image comprehensively and provide detailed, medically accurate guidance in this EXACT format."""
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
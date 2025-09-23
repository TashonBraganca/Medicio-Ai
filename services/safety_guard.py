#!/usr/bin/env python3
"""
MediLens Local - Safety Guard Service
Red-flag symptom detection and safety filtering.
"""

import re
from typing import List, Dict, Set, Tuple
from config import config

class SafetyGuard:
    """Handles safety checks and red-flag detection for medical queries."""
    
    def __init__(self):
        self.red_flag_patterns = self._build_red_flag_patterns()
        self.medical_keywords = self._build_medical_keywords()
    
    def _build_red_flag_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for red-flag symptom detection."""
        return {
            "cardiac": [
                r"chest pain",
                r"heart attack",
                r"crushing pain",
                r"severe chest pressure",
                r"chest tightness with shortness of breath",
                r"chest pain radiating to arm",
                r"chest pain with sweating"
            ],
            "stroke": [
                r"sudden severe headache",
                r"face drooping",
                r"arm weakness",
                r"speech difficulty",
                r"sudden confusion",
                r"sudden trouble seeing",
                r"sudden trouble walking",
                r"sudden loss of coordination"
            ],
            "respiratory": [
                r"severe difficulty breathing",
                r"can't breathe",
                r"gasping for air",
                r"blue lips",
                r"blue fingernails",
                r"choking"
            ],
            "neurological": [
                r"severe head injury",
                r"unconscious",
                r"seizure",
                r"severe confusion",
                r"sudden severe headache",
                r"neck stiffness with fever"
            ],
            "trauma": [
                r"severe bleeding",
                r"heavy bleeding",
                r"broken bone through skin",
                r"severe burn",
                r"poisoning",
                r"overdose"
            ],
            "allergic": [
                r"severe allergic reaction",
                r"anaphylaxis",
                r"throat swelling",
                r"difficulty swallowing",
                r"severe rash all over body"
            ]
        }
    
    def _build_medical_keywords(self) -> Set[str]:
        """Build comprehensive set of medical keywords with extensive vocabulary."""
        return {
            # Symptoms & complaints
            'pain', 'ache', 'hurt', 'hurting', 'sore', 'tender', 'fever', 'headache', 'nausea', 'dizzy', 'fatigue',
            'cough', 'cold', 'flu', 'rash', 'swelling', 'bleeding', 'infection', 'itching', 'burning',
            'vomiting', 'diarrhea', 'constipation', 'shortness of breath', 'weakness', 'numbness', 'tingling',
            'stiff', 'stiffness', 'cramp', 'cramps', 'spasm', 'throbbing', 'pounding', 'stabbing',
            
            # Injuries and wounds - COMPREHENSIVE
            'cut', 'cuts', 'wound', 'wounds', 'injury', 'injuries', 'bruise', 'bruises', 'bruised',
            'burn', 'burns', 'burned', 'burnt', 'scrape', 'scrapes', 'scraped', 'scratch', 'scratched',
            'laceration', 'abrasion', 'trauma', 'sprain', 'strain', 'fracture', 'broken', 'twisted', 
            'swollen', 'scar', 'scars', 'scarred', 'blister', 'blisters', 'bump', 'lumps', 'lump',
            'bite', 'bitten', 'sting', 'stung', 'puncture', 'gash', 'tear', 'torn',
            
            # Animal/insect related - CRITICAL FOR SNAKE BITE
            'snake', 'bite', 'bitten', 'spider', 'bee', 'wasp', 'tick', 'mosquito', 'dog', 'cat',
            'animal', 'insect', 'sting', 'stung', 'venom', 'venomous', 'poison', 'poisonous',
            'rabies', 'tetanus', 'infection', 'swelling', 'reaction', 'allergic',
            
            # Body parts - EXTENSIVE
            'head', 'skull', 'brain', 'face', 'forehead', 'temple', 'cheek', 'jaw', 'chin',
            'eye', 'eyes', 'eyelid', 'eyebrow', 'nose', 'nostril', 'ear', 'ears', 'mouth', 
            'lips', 'tongue', 'tooth', 'teeth', 'gums', 'throat', 'neck',
            'chest', 'breast', 'ribs', 'heart', 'lung', 'lungs', 'stomach', 'abdomen', 'belly',
            'back', 'spine', 'shoulder', 'shoulders', 'arm', 'arms', 'elbow', 'forearm',
            'wrist', 'hand', 'hands', 'finger', 'fingers', 'thumb', 'nail', 'nails',
            'hip', 'hips', 'thigh', 'leg', 'legs', 'knee', 'kneecap', 'shin', 'calf',
            'ankle', 'foot', 'feet', 'toe', 'toes', 'heel', 'sole',
            'skin', 'muscle', 'muscles', 'joint', 'joints', 'bone', 'bones',
            'kidney', 'kidneys', 'liver', 'pancreas', 'bladder', 'genitals',
            
            # Medical terms & procedures
            'diagnosis', 'diagnose', 'diagnosed', 'treatment', 'treat', 'treated', 'medication', 'medicine',
            'drug', 'drugs', 'pill', 'pills', 'tablet', 'capsule', 'therapy', 'surgery', 'operation',
            'hospital', 'clinic', 'doctor', 'physician', 'nurse', 'prescription', 'dosage',
            'lab', 'laboratory', 'test', 'testing', 'result', 'results', 'scan', 'screening',
            'blood pressure', 'blood sugar', 'cholesterol', 'biopsy', 'x-ray', 'mri', 'ct scan',
            'ultrasound', 'examination', 'checkup', 'visit', 'appointment',
            
            # Conditions & diseases - EXPANDED
            'diabetes', 'diabetic', 'hypertension', 'cancer', 'tumor', 'asthma', 'copd',
            'depression', 'anxiety', 'arthritis', 'allergy', 'allergic', 'allergies',
            'infection', 'infected', 'disease', 'disorder', 'syndrome', 'condition',
            'pneumonia', 'bronchitis', 'migraine', 'stroke', 'heart disease', 'heart attack',
            'high blood pressure', 'low blood pressure', 'anemia', 'osteoporosis',
            'fibromyalgia', 'lupus', 'psoriasis', 'eczema', 'dermatitis',
            'gastritis', 'ulcer', 'ibs', 'crohns', 'colitis', 'appendicitis',
            
            # General health & wellness
            'health', 'healthy', 'medical', 'symptom', 'symptoms', 'sick', 'sickness', 'ill', 'illness',
            'wellness', 'fitness', 'exercise', 'diet', 'nutrition', 'weight', 'obesity',
            'vaccine', 'vaccination', 'immunization', 'immunize', 'checkup', 'screening', 'prevention',
            'recovery', 'healing', 'heal', 'better', 'worse', 'improving', 'worsening',
            
            # Emergency & urgency indicators
            'emergency', 'urgent', 'severe', 'serious', 'critical', 'acute', 'chronic',
            'sudden', 'immediate', 'help', 'hospital', 'ambulance', '911', 'doctor',
            
            # Mental health
            'stress', 'stressed', 'worried', 'sad', 'depressed', 'anxious', 'panic', 'mood',
            'sleep', 'sleeping', 'insomnia', 'tired', 'exhausted', 'energy',
            
            # Vital signs & measurements  
            'temperature', 'pulse', 'heartbeat', 'breathing', 'blood', 'urine', 'stool',
            'weight loss', 'weight gain', 'appetite', 'thirst', 'dehydration'
        }
    
    def check_red_flags(self, message: str) -> Dict[str, any]:
        """Check for red-flag symptoms requiring immediate medical attention."""
        message_lower = message.lower()
        detected_flags = {
            "has_red_flags": False,
            "categories": [],
            "matched_symptoms": [],
            "urgency_level": "normal",
            "message": ""
        }
        
        for category, patterns in self.red_flag_patterns.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    detected_flags["has_red_flags"] = True
                    detected_flags["categories"].append(category)
                    detected_flags["matched_symptoms"].append(pattern)
        
        if detected_flags["has_red_flags"]:
            detected_flags["urgency_level"] = "emergency"
            detected_flags["message"] = self._get_emergency_message(detected_flags["categories"])
        
        return detected_flags
    
    def is_medical_query(self, message: str) -> Dict[str, any]:
        """Determine if query is medical/health-related with context awareness."""
        message_lower = message.lower()

        # Check for educational/identification contexts first
        educational_patterns = [
            "what is", "what are", "identify", "identification", "what kind", "what type",
            "species", "looks like", "appears to be", "could be", "might be",
            "picture of", "image of", "photo of", "seen a", "found a", "spotted",
            "educational", "learning", "curious about", "brown snake with", "black spots"
        ]

        is_educational = any(pattern in message_lower for pattern in educational_patterns)

        # Check for actual medical emergency contexts
        emergency_indicators = [
            "bit me", "bitten by", "bite", "attacked", "struck", "hurt", "pain",
            "swelling", "bleeding", "can't breathe", "difficulty breathing",
            "feel sick", "nausea", "emergency", "help", "urgent", "poisoned"
        ]

        has_emergency_context = any(indicator in message_lower for indicator in emergency_indicators)

        # If educational query about animals without emergency context, not medical
        if is_educational and "snake" in message_lower and not has_emergency_context:
            return {
                "is_medical": False,
                "confidence": 0.1,
                "reason": "Educational animal identification query",
                "medical_indicators": 0,
                "context": "educational"
            }

        # Enhanced medical detection with multiple approaches

        # 1. Direct keyword matching (more lenient)
        medical_word_count = sum(1 for keyword in self.medical_keywords if keyword in message_lower)

        # 2. Phrase-based detection for common medical scenarios
        medical_phrases = [
            'snake bite', 'animal bite', 'insect bite', 'bee sting', 'spider bite',
            'hurt', 'pain', 'ache', 'sore', 'injured', 'wound', 'cut', 'bruise',
            'fever', 'sick', 'ill', 'not feeling well', 'feeling unwell',
            'rash', 'swelling', 'bleeding', 'infection', 'burn', 'scratch'
        ]
        phrase_matches = sum(1 for phrase in medical_phrases if phrase in message_lower)

        # 3. Medical context patterns (body parts + action/condition)
        body_parts = ['head', 'arm', 'leg', 'hand', 'foot', 'chest', 'back', 'neck', 'stomach', 'eye', 'ear']
        action_words = ['hurt', 'pain', 'ache', 'sore', 'injured', 'swollen', 'bleeding']
        context_matches = 0
        for body in body_parts:
            for action in action_words:
                if body in message_lower and action in message_lower:
                    context_matches += 1
        
        # Calculate total medical indicators
        total_medical_indicators = medical_word_count + phrase_matches + context_matches
        
        # Much more lenient thresholds
        total_words = len(message_lower.split())
        medical_ratio = total_medical_indicators / max(total_words, 1)
        
        # NEW: Very permissive classification - if ANY medical indicator is found, consider it medical
        is_medical = (
            total_medical_indicators >= 1 or  # Any medical word/phrase found
            medical_ratio >= 0.05 or          # Very low ratio threshold (was 0.1)
            any(urgent_word in message_lower for urgent_word in ['emergency', 'urgent', 'severe', 'serious'])
        )
        
        # Special handling for common false negatives
        if not is_medical:
            # Check for medical scenarios that might not match keywords
            medical_scenarios = [
                'what should i do', 'should i see', 'is this normal', 'how long',
                'when should', 'is it safe', 'recovery time', 'healing time'
            ]
            if any(scenario in message_lower for scenario in medical_scenarios):
                # If combined with any body part or health-related context
                if any(part in message_lower for part in body_parts + ['health', 'medical', 'doctor']):
                    is_medical = True
                    total_medical_indicators = 1
        
        return {
            "is_medical": is_medical,
            "confidence": min(medical_ratio * 100, 100),
            "medical_words_found": medical_word_count,
            "rejection_message": self._get_rejection_message() if not is_medical else None
        }
    
    def validate_response(self, response: str) -> Dict[str, any]:
        """Validate that AI response follows safety guidelines."""
        response_lower = response.lower()
        
        issues = []
        
        # Check for diagnostic language
        diagnostic_words = ['diagnosed', 'diagnose', 'you have', 'you suffer from', 'you are suffering from']
        if any(word in response_lower for word in diagnostic_words):
            issues.append("Contains diagnostic language")
        
        # Check for prescriptive language
        prescriptive_words = ['take this medication', 'prescribe', 'dosage', 'mg', 'ml']
        if any(word in response_lower for word in prescriptive_words):
            issues.append("Contains prescriptive language")
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "requires_disclaimer": False
        }
    
    def _get_emergency_message(self, categories: List[str]) -> str:
        """Get appropriate emergency message based on detected categories."""
        if "cardiac" in categories:
            return "🚨 **EMERGENCY**: These symptoms may indicate a heart attack. Call emergency services (911) immediately or go to the nearest emergency room."
        elif "stroke" in categories:
            return "🚨 **EMERGENCY**: These symptoms may indicate a stroke. Call emergency services (911) immediately. Time is critical for stroke treatment."
        elif "respiratory" in categories:
            return "🚨 **EMERGENCY**: Severe breathing difficulties require immediate medical attention. Call emergency services (911) now."
        elif "neurological" in categories:
            return "🚨 **EMERGENCY**: These neurological symptoms require immediate medical evaluation. Call emergency services (911)."
        elif "trauma" in categories:
            return "🚨 **EMERGENCY**: These injuries require immediate medical attention. Call emergency services (911) or go to the nearest emergency room."
        elif "allergic" in categories:
            return "🚨 **EMERGENCY**: Severe allergic reactions can be life-threatening. Call emergency services (911) immediately."
        else:
            return "🚨 **URGENT**: Your symptoms may require immediate medical attention. Consider calling emergency services or going to the emergency room."
    
    def _get_rejection_message(self) -> str:
        """Get polite rejection message for non-medical queries."""
        return """I'm MediLens, a medical information assistant, and I can only help with health-related questions. 

**I can help with:**
- Symptoms and health concerns
- Understanding medical conditions
- General wellness questions
- Lab results interpretation
- Medication information
- When to seek medical care

**Please ask about:**
- Symptoms you're experiencing
- Medical conditions or treatments
- Health and wellness topics
- Medical test results

How can I assist you with a health-related question today?"""
    

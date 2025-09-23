#!/usr/bin/env python3
"""
CRITICAL MEDICAL SAFETY MODULE
Emergency safety framework to prevent dangerous medical advice
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class CriticalMedicalSafety:
    """EMERGENCY MEDICAL SAFETY FRAMEWORK - Prevents dangerous outputs"""

    def __init__(self):
        # CRITICAL: Life-threatening symptoms requiring immediate 911
        self.emergency_symptoms = {
            "cardiac": [
                "chest pain", "crushing chest pain", "heart attack", "cardiac arrest",
                "severe chest pressure", "chest crushing", "left arm pain with chest",
                "jaw pain with chest", "crushing sensation", "heart stopping"
            ],
            "stroke": [
                "face drooping", "facial droop", "arm weakness sudden", "speech slurred",
                "sudden confusion", "sudden severe headache", "vision loss sudden",
                "numbness one side", "stroke symptoms", "mini stroke"
            ],
            "respiratory": [
                "can't breathe", "cannot breathe", "severe shortness of breath",
                "choking", "blue lips", "blue face", "gasping for air",
                "difficulty breathing severe", "respiratory arrest"
            ],
            "severe_allergic": [
                "anaphylaxis", "severe allergic reaction", "swelling throat",
                "difficulty swallowing sudden", "hives with breathing",
                "allergic shock", "tongue swelling", "airway closing"
            ],
            "trauma": [
                "severe bleeding", "bleeding won't stop", "major bleeding",
                "severe head injury", "unconscious", "loss of consciousness",
                "compound fracture", "bone through skin", "severe burns"
            ],
            "poisoning": [
                "overdose", "poisoning", "toxic ingestion", "swallowed poison",
                "drug overdose", "chemical ingestion", "suicide attempt"
            ]
        }

        # DANGEROUS: Symptoms requiring urgent medical attention (within hours)
        self.urgent_symptoms = [
            "severe pain", "high fever", "persistent vomiting", "severe headache",
            "chest discomfort", "difficulty breathing", "severe dizziness",
            "blood in urine", "blood in stool", "severe abdominal pain",
            "diabetic", "seizure", "pregnancy bleeding", "severe infection"
        ]

        # CONTRAINDICATED: Medical advice that should NEVER be given
        self.dangerous_advice = [
            "take more medication", "increase dose", "stop medication",
            "don't see doctor", "wait it out", "ignore symptoms",
            "self-medicate", "home surgery", "delay treatment"
        ]

        # Medical disclaimers
        self.critical_disclaimer = """
🚨 **CRITICAL MEDICAL EMERGENCY ALERT** 🚨

CALL 911 IMMEDIATELY OR GO TO NEAREST EMERGENCY ROOM

This AI detected potential life-threatening symptoms that require immediate professional medical intervention.

DO NOT DELAY - SEEK EMERGENCY MEDICAL CARE NOW
        """

        self.urgent_disclaimer = """
⚠️ **URGENT MEDICAL ATTENTION REQUIRED** ⚠️

Contact your healthcare provider immediately or visit urgent care within 2-4 hours.

These symptoms may indicate a serious medical condition requiring prompt evaluation.
        """

        self.general_disclaimer = """
📋 **IMPORTANT MEDICAL DISCLAIMER**

This AI provides general health information only and is NOT a substitute for professional medical advice, diagnosis, or treatment.

✓ Always consult qualified healthcare providers for medical decisions
✓ In emergencies, call 911 or emergency services immediately
✓ Do not ignore professional medical advice based on AI information
✓ This tool is for educational purposes only

**AI Accuracy Limitation**: This system uses general-purpose language models without specialized medical training and may provide inaccurate information.
        """

    def analyze_medical_safety(self, user_input: str, ai_response: str) -> Dict[str, any]:
        """
        CRITICAL SAFETY ANALYSIS - Prevents dangerous medical advice

        Returns:
            - safety_level: "emergency", "urgent", "caution", "safe"
            - intervention_required: bool
            - safety_message: str
            - blocked_response: bool
        """
        safety_analysis = {
            "safety_level": "safe",
            "intervention_required": False,
            "safety_message": "",
            "blocked_response": False,
            "emergency_detected": False,
            "urgent_detected": False,
            "dangerous_advice_detected": False
        }

        # STEP 1: Emergency symptom detection
        emergency_detected = self._detect_emergency_symptoms(user_input)
        if emergency_detected:
            safety_analysis.update({
                "safety_level": "emergency",
                "intervention_required": True,
                "safety_message": self.critical_disclaimer,
                "blocked_response": True,
                "emergency_detected": True,
                "emergency_type": emergency_detected
            })
            return safety_analysis

        # STEP 2: Urgent symptom detection
        urgent_detected = self._detect_urgent_symptoms(user_input)
        if urgent_detected:
            safety_analysis.update({
                "safety_level": "urgent",
                "intervention_required": True,
                "safety_message": self.urgent_disclaimer,
                "urgent_detected": True
            })

        # STEP 3: Dangerous advice detection in AI response
        dangerous_advice = self._detect_dangerous_advice(ai_response)
        if dangerous_advice:
            safety_analysis.update({
                "safety_level": "urgent",
                "intervention_required": True,
                "blocked_response": True,
                "dangerous_advice_detected": True,
                "dangerous_elements": dangerous_advice
            })

        # STEP 4: General medical disclaimer (always required)
        if safety_analysis["safety_level"] == "safe":
            safety_analysis["safety_message"] = self.general_disclaimer

        return safety_analysis

    def _detect_emergency_symptoms(self, text: str) -> Optional[str]:
        """Detect life-threatening emergency symptoms with context awareness"""
        text_lower = text.lower()

        # Check for non-emergency contexts first
        educational_patterns = [
            "what is", "what are", "identify", "identification", "brown snake with",
            "type of snake", "species", "what kind", "looks like", "appears to be",
            "could be", "might be", "picture of", "image of", "photo of", "seen a",
            "found a", "spotted", "educational", "learning", "curious about"
        ]

        # If this appears to be educational/identification request, don't treat as emergency
        if any(pattern in text_lower for pattern in educational_patterns):
            # Extra check for snake bite context
            bite_indicators = ["bite", "bitten", "bit me", "bit by", "attacked", "struck"]
            if not any(indicator in text_lower for indicator in bite_indicators):
                logger.info("Educational/identification query detected - not emergency")
                return None

        # Check for actual bite or medical emergency context
        emergency_contexts = [
            "bit me", "bitten by", "bite", "attacked", "struck", "hurt", "pain",
            "swelling", "bleeding", "can't breathe", "difficulty breathing",
            "feel sick", "nausea", "emergency", "help", "urgent"
        ]

        has_emergency_context = any(context in text_lower for context in emergency_contexts)

        for category, symptoms in self.emergency_symptoms.items():
            for symptom in symptoms:
                if symptom in text_lower:
                    # For snake-related queries, require emergency context
                    if "snake" in text_lower and not has_emergency_context:
                        logger.info(f"Snake mentioned but no emergency context: {symptom}")
                        continue

                    logger.critical(f"EMERGENCY SYMPTOM DETECTED: {symptom} (Category: {category})")
                    return category
        return None

    def _detect_urgent_symptoms(self, text: str) -> bool:
        """Detect symptoms requiring urgent medical attention"""
        text_lower = text.lower()

        for symptom in self.urgent_symptoms:
            if symptom in text_lower:
                logger.warning(f"URGENT SYMPTOM DETECTED: {symptom}")
                return True
        return False

    def _detect_dangerous_advice(self, response: str) -> List[str]:
        """Detect dangerous medical advice in AI response"""
        response_lower = response.lower()
        detected_dangerous = []

        for advice in self.dangerous_advice:
            if advice in response_lower:
                detected_dangerous.append(advice)
                logger.error(f"DANGEROUS ADVICE DETECTED: {advice}")

        return detected_dangerous

    def create_safe_response(self, safety_analysis: Dict, original_response: str = "") -> str:
        """Create medically safe response based on safety analysis"""

        if safety_analysis["emergency_detected"]:
            return f"""
{self.critical_disclaimer}

**Emergency Type Detected**: {safety_analysis.get('emergency_type', 'Critical symptoms')}

**IMMEDIATE ACTIONS:**
1. CALL 911 NOW
2. Do not drive yourself - call ambulance
3. If unconscious, call emergency services immediately
4. Stay with the person until help arrives

This AI is not qualified to provide emergency medical guidance.
            """

        elif safety_analysis["urgent_detected"]:
            return f"""
{self.urgent_disclaimer}

**Next Steps:**
1. Contact your healthcare provider immediately
2. Visit urgent care or emergency room if provider unavailable
3. Do not wait for symptoms to worsen
4. Monitor symptoms closely

{self.general_disclaimer}
            """

        elif safety_analysis["dangerous_advice_detected"]:
            return f"""
⚠️ **RESPONSE BLOCKED FOR SAFETY**

This AI detected potentially dangerous medical advice in the response.

**Instead:**
- Consult qualified healthcare providers for medical decisions
- Never stop medications without physician guidance
- Seek professional medical evaluation for concerning symptoms

{self.general_disclaimer}
            """

        else:
            # Safe response - add disclaimer
            return f"""
{original_response}

---

{self.general_disclaimer}
            """

    def validate_medical_response(self, user_input: str, ai_response: str) -> Tuple[bool, str]:
        """
        MAIN VALIDATION FUNCTION

        Returns:
            - is_safe: bool (True if response can be shown)
            - final_response: str (safe response to display)
        """

        # Analyze safety
        safety_analysis = self.analyze_medical_safety(user_input, ai_response)

        # Log safety event
        self._log_safety_event(user_input, ai_response, safety_analysis)

        # Create safe response
        if safety_analysis["blocked_response"]:
            # Response is blocked - show safety message only
            safe_response = self.create_safe_response(safety_analysis)
            return False, safe_response
        else:
            # Response is allowed with disclaimer
            safe_response = self.create_safe_response(safety_analysis, ai_response)
            return True, safe_response

    def _log_safety_event(self, user_input: str, ai_response: str, safety_analysis: Dict):
        """Log medical safety events for monitoring"""
        timestamp = datetime.now().isoformat()

        safety_event = {
            "timestamp": timestamp,
            "safety_level": safety_analysis["safety_level"],
            "emergency_detected": safety_analysis["emergency_detected"],
            "urgent_detected": safety_analysis["urgent_detected"],
            "dangerous_advice_detected": safety_analysis["dangerous_advice_detected"],
            "user_input_hash": hash(user_input) % 10000,  # Privacy-preserving
            "intervention_required": safety_analysis["intervention_required"]
        }

        # Log critical events
        if safety_analysis["emergency_detected"]:
            logger.critical(f"MEDICAL EMERGENCY DETECTED: {safety_event}")
        elif safety_analysis["urgent_detected"]:
            logger.warning(f"URGENT MEDICAL ATTENTION REQUIRED: {safety_event}")
        elif safety_analysis["dangerous_advice_detected"]:
            logger.error(f"DANGEROUS ADVICE BLOCKED: {safety_event}")

# Global safety instance
medical_safety = CriticalMedicalSafety()
#!/usr/bin/env python3
"""
MEDICAL KNOWLEDGE DATABASE & MODEL ENHANCEMENT
Specialized medical knowledge integration to improve model accuracy
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class MedicalKnowledgeBase:
    """Medical knowledge database for validating and enhancing AI responses"""

    def __init__(self):
        # CRITICAL MEDICAL CONDITIONS DATABASE
        self.medical_conditions = {
            "cardiac": {
                "myocardial_infarction": {
                    "symptoms": ["chest pain", "crushing chest pain", "left arm pain", "jaw pain", "shortness of breath"],
                    "red_flags": ["severe chest pain", "crushing sensation", "pain radiating to arm"],
                    "immediate_action": "CALL 911 IMMEDIATELY",
                    "triage_level": "emergency"
                },
                "unstable_angina": {
                    "symptoms": ["chest discomfort", "chest tightness", "exertional chest pain"],
                    "red_flags": ["rest chest pain", "increasing frequency", "new onset"],
                    "immediate_action": "Seek emergency medical care",
                    "triage_level": "urgent"
                }
            },
            "neurological": {
                "stroke": {
                    "symptoms": ["face drooping", "arm weakness", "speech difficulty", "sudden confusion"],
                    "red_flags": ["sudden onset", "severe headache", "vision loss"],
                    "immediate_action": "CALL 911 IMMEDIATELY",
                    "triage_level": "emergency"
                },
                "tia": {
                    "symptoms": ["temporary weakness", "temporary speech problems", "temporary vision loss"],
                    "red_flags": ["sudden onset", "multiple episodes"],
                    "immediate_action": "Seek immediate medical attention",
                    "triage_level": "urgent"
                }
            },
            "respiratory": {
                "pulmonary_embolism": {
                    "symptoms": ["sudden shortness of breath", "chest pain", "coughing up blood"],
                    "red_flags": ["sudden onset", "recent surgery", "leg swelling"],
                    "immediate_action": "CALL 911 IMMEDIATELY",
                    "triage_level": "emergency"
                },
                "pneumonia": {
                    "symptoms": ["cough", "fever", "shortness of breath", "chest pain"],
                    "red_flags": ["high fever", "severe breathing difficulty", "confusion"],
                    "immediate_action": "Seek medical attention within 24 hours",
                    "triage_level": "urgent"
                }
            }
        }

        # DRUG INTERACTION DATABASE (Critical subset)
        self.drug_interactions = {
            "warfarin": {
                "dangerous_interactions": ["aspirin", "ibuprofen", "acetaminophen_high_dose"],
                "monitoring_required": ["inr_levels", "bleeding_signs"],
                "contraindications": ["active_bleeding", "pregnancy"]
            },
            "metformin": {
                "dangerous_interactions": ["contrast_dye", "alcohol_excess"],
                "monitoring_required": ["kidney_function", "lactic_acidosis_signs"],
                "contraindications": ["kidney_disease", "heart_failure"]
            }
        }

        # NORMAL LAB VALUES DATABASE
        self.normal_ranges = {
            "glucose_fasting": {"min": 70, "max": 100, "unit": "mg/dL", "critical_high": 400, "critical_low": 50},
            "cholesterol_total": {"min": 0, "max": 200, "unit": "mg/dL", "critical_high": 300},
            "blood_pressure_systolic": {"min": 90, "max": 120, "unit": "mmHg", "critical_high": 180, "critical_low": 70},
            "hemoglobin": {"min": 12.0, "max": 16.0, "unit": "g/dL", "critical_low": 7.0, "critical_high": 20.0},
            "heart_rate": {"min": 60, "max": 100, "unit": "bpm", "critical_high": 150, "critical_low": 40}
        }

        # MEDICAL SPECIALTIES REFERRAL GUIDE
        self.specialist_referrals = {
            "cardiology": ["chest pain", "heart rhythm", "blood pressure", "cholesterol"],
            "neurology": ["headache", "seizure", "stroke", "memory", "numbness"],
            "pulmonology": ["breathing", "cough", "lung", "asthma", "copd"],
            "gastroenterology": ["stomach pain", "nausea", "diarrhea", "constipation"],
            "endocrinology": ["diabetes", "thyroid", "hormone", "metabolism"],
            "emergency_medicine": ["severe pain", "trauma", "poisoning", "overdose"]
        }

    def enhance_medical_response(self, user_query: str, ai_response: str) -> Dict[str, any]:
        """
        Enhance AI response with medical knowledge validation

        Returns enhanced response with:
        - Medical accuracy validation
        - Specialist referral recommendations
        - Clinical context
        - Evidence-based information
        """
        enhancement = {
            "original_response": ai_response,
            "enhanced_response": ai_response,
            "medical_validation": True,
            "accuracy_score": 0.0,
            "clinical_notes": [],
            "specialist_referral": None,
            "red_flag_detected": False,
            "knowledge_gaps": []
        }

        try:
            # Step 1: Validate against medical knowledge
            validation_result = self._validate_medical_accuracy(user_query, ai_response)
            enhancement.update(validation_result)

            # Step 2: Add clinical context
            clinical_context = self._add_clinical_context(user_query)
            enhancement["clinical_context"] = clinical_context

            # Step 3: Recommend specialist if needed
            specialist = self._recommend_specialist(user_query)
            enhancement["specialist_referral"] = specialist

            # Step 4: Validate lab values if present
            lab_validation = self._validate_lab_values(ai_response)
            enhancement["lab_validation"] = lab_validation

            # Step 5: Check drug interactions
            drug_check = self._check_drug_interactions(user_query, ai_response)
            enhancement["drug_interaction_check"] = drug_check

            # Step 6: Generate enhanced response
            enhancement["enhanced_response"] = self._generate_enhanced_response(
                ai_response, enhancement
            )

        except Exception as e:
            logger.error(f"Error enhancing medical response: {str(e)}")
            enhancement["error"] = str(e)

        return enhancement

    def _validate_medical_accuracy(self, query: str, response: str) -> Dict[str, any]:
        """Validate response against medical knowledge base"""
        validation = {
            "medical_validation": True,
            "accuracy_score": 0.7,  # Base score for general models
            "knowledge_gaps": [],
            "medical_errors": []
        }

        query_lower = query.lower()
        response_lower = response.lower()

        # Check for dangerous advice patterns
        dangerous_patterns = [
            "ignore symptoms", "don't see doctor", "wait it out",
            "increase medication", "stop taking", "home surgery"
        ]

        for pattern in dangerous_patterns:
            if pattern in response_lower:
                validation["medical_errors"].append(f"Dangerous advice detected: {pattern}")
                validation["accuracy_score"] -= 0.3

        # Check for missing critical information
        if any(symptom in query_lower for symptom in ["chest pain", "shortness of breath"]):
            if "emergency" not in response_lower and "911" not in response_lower:
                validation["knowledge_gaps"].append("Missing emergency guidance for serious symptoms")
                validation["accuracy_score"] -= 0.2

        return validation

    def _add_clinical_context(self, query: str) -> Dict[str, any]:
        """Add relevant clinical context based on query"""
        context = {
            "condition_detected": None,
            "clinical_guidelines": [],
            "differential_diagnosis": [],
            "risk_factors": []
        }

        query_lower = query.lower()

        # Match to known conditions
        for category, conditions in self.medical_conditions.items():
            for condition, details in conditions.items():
                if any(symptom in query_lower for symptom in details["symptoms"]):
                    context["condition_detected"] = condition
                    context["clinical_guidelines"] = [details["immediate_action"]]
                    context["triage_level"] = details["triage_level"]
                    break

        return context

    def _recommend_specialist(self, query: str) -> Optional[str]:
        """Recommend appropriate medical specialist"""
        query_lower = query.lower()

        for specialty, keywords in self.specialist_referrals.items():
            if any(keyword in query_lower for keyword in keywords):
                return specialty

        return None

    def _validate_lab_values(self, response: str) -> Dict[str, any]:
        """Validate any lab values mentioned in response"""
        validation = {
            "values_found": [],
            "normal_ranges_provided": False,
            "critical_values_detected": []
        }

        # Extract numerical values that might be lab results
        value_pattern = r'(\w+)[:\s]*(\d+\.?\d*)\s*([a-zA-Z/]+)?'
        matches = re.findall(value_pattern, response.lower())

        for test_name, value, unit in matches:
            if test_name in self.normal_ranges:
                normal_range = self.normal_ranges[test_name]
                value_float = float(value)

                validation["values_found"].append({
                    "test": test_name,
                    "value": value_float,
                    "unit": unit or normal_range["unit"],
                    "normal_range": f"{normal_range['min']}-{normal_range['max']} {normal_range['unit']}"
                })

                # Check for critical values
                if ("critical_high" in normal_range and value_float > normal_range["critical_high"]) or \
                   ("critical_low" in normal_range and value_float < normal_range["critical_low"]):
                    validation["critical_values_detected"].append(test_name)

        return validation

    def _check_drug_interactions(self, query: str, response: str) -> Dict[str, any]:
        """Check for potential drug interactions"""
        interaction_check = {
            "medications_mentioned": [],
            "interactions_detected": [],
            "warnings": []
        }

        text = (query + " " + response).lower()

        # Extract mentioned medications
        for drug in self.drug_interactions.keys():
            if drug in text:
                interaction_check["medications_mentioned"].append(drug)

        # Check for dangerous combinations
        if len(interaction_check["medications_mentioned"]) > 1:
            interaction_check["warnings"].append(
                "Multiple medications mentioned - consult pharmacist for interaction check"
            )

        return interaction_check

    def _generate_enhanced_response(self, original_response: str, enhancement: Dict) -> str:
        """Generate enhanced response with medical knowledge"""
        enhanced = original_response

        # Add clinical context if available
        if enhancement.get("clinical_context", {}).get("condition_detected"):
            condition = enhancement["clinical_context"]["condition_detected"]
            enhanced += f"\n\n**Clinical Note**: Symptoms may be consistent with {condition.replace('_', ' ')}."

        # Add specialist referral recommendation
        if enhancement.get("specialist_referral"):
            specialist = enhancement["specialist_referral"]
            enhanced += f"\n\n**Specialist Recommendation**: Consider consultation with {specialist}."

        # Add lab value context
        lab_validation = enhancement.get("lab_validation", {})
        if lab_validation.get("values_found"):
            enhanced += "\n\n**Lab Value Reference**:"
            for lab in lab_validation["values_found"]:
                enhanced += f"\n• {lab['test'].title()}: Normal range {lab['normal_range']}"

        # Add drug interaction warnings
        drug_check = enhancement.get("drug_interaction_check", {})
        if drug_check.get("warnings"):
            enhanced += "\n\n**Medication Safety**:"
            for warning in drug_check["warnings"]:
                enhanced += f"\n• {warning}"

        return enhanced

# Global medical knowledge instance
medical_knowledge = MedicalKnowledgeBase()
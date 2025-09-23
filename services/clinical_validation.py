#!/usr/bin/env python3
"""
CLINICAL VALIDATION & OVERSIGHT SYSTEM
Implements medical validation framework with clinical oversight protocols
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Clinical validation severity levels"""
    CRITICAL = "critical"      # Requires immediate clinical review
    HIGH = "high"             # Requires clinical review within 24h
    MODERATE = "moderate"     # Requires periodic clinical audit
    LOW = "low"              # Standard AI response acceptable

class ClinicalSpecialty(Enum):
    """Medical specialties for referral validation"""
    EMERGENCY_MEDICINE = "emergency_medicine"
    CARDIOLOGY = "cardiology"
    NEUROLOGY = "neurology"
    PULMONOLOGY = "pulmonology"
    GASTROENTEROLOGY = "gastroenterology"
    ENDOCRINOLOGY = "endocrinology"
    DERMATOLOGY = "dermatology"
    PSYCHIATRY = "psychiatry"
    GENERAL_PRACTICE = "general_practice"

@dataclass
class ValidationResult:
    """Clinical validation result structure"""
    validation_level: ValidationLevel
    clinical_accuracy_score: float
    evidence_quality: str
    specialist_required: Optional[ClinicalSpecialty]
    clinical_notes: List[str]
    contraindications: List[str]
    follow_up_required: bool
    liability_risk: str
    validation_timestamp: datetime

class ClinicalValidationSystem:
    """Clinical validation and oversight system for medical AI responses"""

    def __init__(self):
        # CLINICAL VALIDATION CRITERIA
        self.high_risk_conditions = {
            "cardiovascular": {
                "symptoms": ["chest pain", "heart attack", "cardiac arrest", "arrhythmia"],
                "validation_level": ValidationLevel.CRITICAL,
                "specialist": ClinicalSpecialty.CARDIOLOGY,
                "max_response_time": timedelta(minutes=15)
            },
            "neurological": {
                "symptoms": ["stroke", "seizure", "head injury", "loss of consciousness"],
                "validation_level": ValidationLevel.CRITICAL,
                "specialist": ClinicalSpecialty.NEUROLOGY,
                "max_response_time": timedelta(minutes=15)
            },
            "respiratory": {
                "symptoms": ["severe shortness of breath", "respiratory distress", "choking"],
                "validation_level": ValidationLevel.CRITICAL,
                "specialist": ClinicalSpecialty.PULMONOLOGY,
                "max_response_time": timedelta(minutes=15)
            },
            "trauma": {
                "symptoms": ["severe bleeding", "major trauma", "burns", "fractures"],
                "validation_level": ValidationLevel.CRITICAL,
                "specialist": ClinicalSpecialty.EMERGENCY_MEDICINE,
                "max_response_time": timedelta(minutes=10)
            }
        }

        # EVIDENCE-BASED MEDICAL GUIDELINES
        self.clinical_guidelines = {
            "chest_pain_evaluation": {
                "red_flags": ["crushing pain", "radiation to arm", "sweating", "nausea"],
                "immediate_actions": ["ECG", "cardiac enzymes", "aspirin if no contraindications"],
                "specialist_referral": ClinicalSpecialty.CARDIOLOGY,
                "evidence_level": "A"  # High-quality evidence
            },
            "stroke_recognition": {
                "red_flags": ["FAST criteria", "sudden onset", "focal deficits"],
                "immediate_actions": ["time of onset", "CT head", "thrombolytics consideration"],
                "specialist_referral": ClinicalSpecialty.NEUROLOGY,
                "evidence_level": "A"
            },
            "respiratory_distress": {
                "red_flags": ["oxygen saturation", "respiratory rate", "accessory muscles"],
                "immediate_actions": ["oxygen therapy", "positioning", "bronchodilators"],
                "specialist_referral": ClinicalSpecialty.PULMONOLOGY,
                "evidence_level": "A"
            }
        }

        # CONTRAINDICATION DATABASE
        self.contraindications = {
            "aspirin": {
                "absolute": ["active bleeding", "severe liver disease", "allergy"],
                "relative": ["pregnancy", "asthma", "peptic ulcer disease"],
                "interactions": ["warfarin", "methotrexate"]
            },
            "thrombolytics": {
                "absolute": ["active bleeding", "recent surgery", "stroke within 3 months"],
                "relative": ["age >75", "pregnancy", "hypertension"],
                "time_window": "4.5 hours for stroke"
            }
        }

        # CLINICAL DECISION SUPPORT RULES
        self.decision_rules = {
            "ottawa_ankle_rules": {
                "criteria": ["bone tenderness", "inability to bear weight"],
                "recommendation": "X-ray indicated if criteria met",
                "sensitivity": "98-99%"
            },
            "wells_score_pe": {
                "criteria": ["clinical signs of DVT", "PE likely", "heart rate >100"],
                "recommendation": "D-dimer or CT-PA based on score",
                "validation": "Clinically validated"
            }
        }

        # MEDICATION SAFETY DATABASE
        self.medication_safety = {
            "high_alert_medications": [
                "insulin", "heparin", "warfarin", "chemotherapy", "opioids"
            ],
            "pregnancy_categories": {
                "category_x": ["warfarin", "thalidomide", "methotrexate"],
                "category_d": ["phenytoin", "valproic acid", "lithium"]
            },
            "geriatric_considerations": {
                "beers_criteria": ["diphenhydramine", "amitriptyline", "diazepam"],
                "dose_adjustments": ["digoxin", "lithium", "warfarin"]
            }
        }

    def validate_clinical_response(self,
                                 user_query: str,
                                 ai_response: str,
                                 patient_context: Optional[Dict] = None) -> ValidationResult:
        """
        Comprehensive clinical validation of AI medical response

        Args:
            user_query: Patient's medical query
            ai_response: AI-generated medical response
            patient_context: Additional patient information (age, medications, etc.)

        Returns:
            ValidationResult with clinical assessment
        """

        # Step 1: Determine validation level
        validation_level = self._assess_validation_level(user_query)

        # Step 2: Clinical accuracy assessment
        accuracy_score = self._assess_clinical_accuracy(user_query, ai_response)

        # Step 3: Evidence quality evaluation
        evidence_quality = self._evaluate_evidence_quality(ai_response)

        # Step 4: Specialist referral determination
        specialist_required = self._determine_specialist_referral(user_query)

        # Step 5: Generate clinical notes
        clinical_notes = self._generate_clinical_notes(user_query, ai_response)

        # Step 6: Check contraindications
        contraindications = self._check_contraindications(ai_response, patient_context)

        # Step 7: Determine follow-up requirements
        follow_up_required = self._assess_follow_up_needs(user_query, validation_level)

        # Step 8: Assess liability risk
        liability_risk = self._assess_liability_risk(validation_level, accuracy_score)

        return ValidationResult(
            validation_level=validation_level,
            clinical_accuracy_score=accuracy_score,
            evidence_quality=evidence_quality,
            specialist_required=specialist_required,
            clinical_notes=clinical_notes,
            contraindications=contraindications,
            follow_up_required=follow_up_required,
            liability_risk=liability_risk,
            validation_timestamp=datetime.now()
        )

    def _assess_validation_level(self, query: str) -> ValidationLevel:
        """Determine required clinical validation level"""
        query_lower = query.lower()

        # Check for critical conditions requiring immediate validation
        for condition, details in self.high_risk_conditions.items():
            if any(symptom in query_lower for symptom in details["symptoms"]):
                return details["validation_level"]

        # Check for high-risk keywords
        high_risk_keywords = [
            "severe", "sudden", "emergency", "urgent", "critical",
            "bleeding", "pain", "difficulty breathing", "unconscious"
        ]

        if any(keyword in query_lower for keyword in high_risk_keywords):
            return ValidationLevel.HIGH

        # Check for moderate risk conditions
        moderate_risk_keywords = [
            "chronic", "medication", "treatment", "diagnosis", "symptoms"
        ]

        if any(keyword in query_lower for keyword in moderate_risk_keywords):
            return ValidationLevel.MODERATE

        return ValidationLevel.LOW

    def _assess_clinical_accuracy(self, query: str, response: str) -> float:
        """Assess clinical accuracy of AI response"""
        accuracy_score = 0.5  # Base score

        query_lower = query.lower()
        response_lower = response.lower()

        # Positive accuracy indicators
        accuracy_positives = [
            "consult healthcare provider",
            "seek medical attention",
            "emergency services",
            "follow up with doctor",
            "evidence-based",
            "clinical guidelines"
        ]

        # Negative accuracy indicators
        accuracy_negatives = [
            "definitely",
            "certainly",
            "guaranteed",
            "cure",
            "never see doctor",
            "ignore symptoms"
        ]

        # Score based on positive indicators
        positive_count = sum(1 for indicator in accuracy_positives if indicator in response_lower)
        accuracy_score += positive_count * 0.1

        # Penalize negative indicators
        negative_count = sum(1 for indicator in accuracy_negatives if indicator in response_lower)
        accuracy_score -= negative_count * 0.2

        # Bonus for including appropriate disclaimers
        if "not a substitute for professional medical advice" in response_lower:
            accuracy_score += 0.1

        return min(1.0, max(0.0, accuracy_score))

    def _evaluate_evidence_quality(self, response: str) -> str:
        """Evaluate quality of evidence in AI response"""
        response_lower = response.lower()

        # High-quality evidence indicators
        high_quality = [
            "clinical guidelines", "evidence-based", "peer-reviewed",
            "randomized controlled trial", "meta-analysis", "systematic review"
        ]

        # Moderate-quality evidence indicators
        moderate_quality = [
            "clinical study", "research shows", "studies indicate",
            "medical literature", "clinical experience"
        ]

        # Low-quality evidence indicators
        low_quality = [
            "anecdotal", "case report", "expert opinion",
            "preliminary study", "small study"
        ]

        if any(indicator in response_lower for indicator in high_quality):
            return "high"
        elif any(indicator in response_lower for indicator in moderate_quality):
            return "moderate"
        elif any(indicator in response_lower for indicator in low_quality):
            return "low"
        else:
            return "unknown"

    def _determine_specialist_referral(self, query: str) -> Optional[ClinicalSpecialty]:
        """Determine if specialist referral is needed"""
        query_lower = query.lower()

        # Map symptoms to specialties
        specialty_mapping = {
            ClinicalSpecialty.CARDIOLOGY: [
                "chest pain", "heart", "cardiac", "blood pressure", "cholesterol"
            ],
            ClinicalSpecialty.NEUROLOGY: [
                "headache", "seizure", "stroke", "memory", "numbness", "weakness"
            ],
            ClinicalSpecialty.PULMONOLOGY: [
                "breathing", "cough", "lung", "asthma", "copd", "shortness of breath"
            ],
            ClinicalSpecialty.EMERGENCY_MEDICINE: [
                "emergency", "urgent", "severe", "trauma", "critical", "life-threatening"
            ]
        }

        for specialty, keywords in specialty_mapping.items():
            if any(keyword in query_lower for keyword in keywords):
                return specialty

        return ClinicalSpecialty.GENERAL_PRACTICE

    def _generate_clinical_notes(self, query: str, response: str) -> List[str]:
        """Generate clinical validation notes"""
        notes = []

        query_lower = query.lower()
        response_lower = response.lower()

        # Check for missing critical information
        if "chest pain" in query_lower:
            if "ecg" not in response_lower and "electrocardiogram" not in response_lower:
                notes.append("Consider recommending ECG evaluation for chest pain")

        if "shortness of breath" in query_lower:
            if "oxygen saturation" not in response_lower:
                notes.append("Consider oxygen saturation assessment")

        # Check for medication recommendations
        if "medication" in response_lower or "drug" in response_lower:
            notes.append("Medication recommendations require clinical validation")

        # Check for diagnostic recommendations
        if any(test in response_lower for test in ["x-ray", "ct scan", "mri", "blood test"]):
            notes.append("Diagnostic test recommendations should be clinically validated")

        return notes

    def _check_contraindications(self, response: str, patient_context: Optional[Dict]) -> List[str]:
        """Check for potential contraindications"""
        contraindications_found = []
        response_lower = response.lower()

        # Check medication contraindications
        if "aspirin" in response_lower:
            contraindications_found.append("Verify no aspirin contraindications (bleeding, allergy)")

        if patient_context:
            age = patient_context.get("age")
            if age and age > 65:
                contraindications_found.append("Consider geriatric dosing adjustments")

            pregnancy = patient_context.get("pregnant")
            if pregnancy:
                contraindications_found.append("Verify pregnancy safety for all recommendations")

        return contraindications_found

    def _assess_follow_up_needs(self, query: str, validation_level: ValidationLevel) -> bool:
        """Determine if clinical follow-up is required"""
        return validation_level in [ValidationLevel.CRITICAL, ValidationLevel.HIGH]

    def _assess_liability_risk(self, validation_level: ValidationLevel, accuracy_score: float) -> str:
        """Assess potential liability risk"""
        if validation_level == ValidationLevel.CRITICAL:
            return "high"
        elif validation_level == ValidationLevel.HIGH or accuracy_score < 0.7:
            return "moderate"
        else:
            return "low"

    def generate_clinical_override_recommendations(self, validation_result: ValidationResult) -> Dict[str, Any]:
        """Generate recommendations for clinical oversight"""
        recommendations = {
            "immediate_actions": [],
            "clinical_review_required": False,
            "escalation_needed": False,
            "documentation_requirements": [],
            "quality_assurance_flags": []
        }

        if validation_result.validation_level == ValidationLevel.CRITICAL:
            recommendations["immediate_actions"].append("Immediate clinical review required")
            recommendations["clinical_review_required"] = True
            recommendations["escalation_needed"] = True

        if validation_result.clinical_accuracy_score < 0.6:
            recommendations["quality_assurance_flags"].append("Low accuracy score - review AI response")

        if validation_result.contraindications:
            recommendations["immediate_actions"].append("Review contraindications with clinical staff")

        if validation_result.specialist_required:
            recommendations["documentation_requirements"].append(
                f"Document specialist referral to {validation_result.specialist_required.value}"
            )

        return recommendations

# Global clinical validation instance
clinical_validator = ClinicalValidationSystem()
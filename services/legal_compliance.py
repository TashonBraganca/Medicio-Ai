#!/usr/bin/env python3
"""
LEGAL COMPLIANCE & LIABILITY PROTECTION FRAMEWORK
Comprehensive legal compliance for medical AI applications
"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import hashlib
import uuid

logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """Legal compliance requirement levels"""
    CRITICAL = "critical"      # Full legal review required
    HIGH = "high"             # Legal consultation recommended
    MODERATE = "moderate"     # Standard compliance monitoring
    LOW = "low"              # Basic compliance sufficient

class LiabilityRisk(Enum):
    """Liability risk assessment levels"""
    EXTREME = "extreme"       # Legal intervention required
    HIGH = "high"            # Risk mitigation needed
    MODERATE = "moderate"    # Standard precautions
    LOW = "low"             # Minimal risk

@dataclass
class ComplianceRecord:
    """Legal compliance audit record"""
    record_id: str
    timestamp: datetime
    compliance_level: ComplianceLevel
    liability_risk: LiabilityRisk
    user_query_hash: str
    ai_response_hash: str
    disclaimers_present: bool
    emergency_protocols_followed: bool
    professional_supervision: bool
    audit_trail: List[str]
    compliance_notes: List[str]
    legal_recommendations: List[str]

class LegalComplianceFramework:
    """Legal compliance and liability protection system"""

    def __init__(self):
        # FDA MEDICAL DEVICE REGULATIONS (21 CFR Part 820)
        self.fda_requirements = {
            "medical_device_software": {
                "class_i": "Low risk - 510(k) exempt",
                "class_ii": "Moderate risk - 510(k) required",
                "class_iii": "High risk - PMA required"
            },
            "clinical_decision_support": {
                "requirements": [
                    "Clinical validation required",
                    "FDA premarket notification",
                    "Quality system regulation compliance",
                    "Adverse event reporting"
                ]
            }
        }

        # HIPAA COMPLIANCE REQUIREMENTS
        self.hipaa_requirements = {
            "protected_health_information": {
                "identifiers": [
                    "names", "addresses", "dates", "phone_numbers",
                    "email_addresses", "ssn", "medical_record_numbers"
                ],
                "safeguards": [
                    "access_controls", "audit_controls", "integrity",
                    "person_authentication", "transmission_security"
                ]
            },
            "minimum_necessary": "Only minimum PHI necessary for purpose",
            "business_associate": "BAA required for third-party services"
        }

        # MALPRACTICE LIABILITY PROTECTIONS
        self.malpractice_protections = {
            "required_disclaimers": [
                "Not a substitute for professional medical advice",
                "For informational purposes only",
                "Always consult qualified healthcare providers",
                "In emergencies, contact emergency services immediately",
                "This tool has limitations and may provide inaccurate information"
            ],
            "informed_consent_elements": [
                "Nature and purpose of AI assistance",
                "Limitations and potential risks",
                "Alternative options available",
                "Right to discontinue use"
            ],
            "professional_oversight": [
                "Licensed healthcare provider supervision",
                "Clinical validation of AI recommendations",
                "Human-in-the-loop decision making",
                "Regular audit and quality assurance"
            ]
        }

        # STATE MEDICAL PRACTICE LAWS
        self.medical_practice_laws = {
            "unlicensed_practice": {
                "prohibited_activities": [
                    "Medical diagnosis", "Treatment recommendations",
                    "Prescription advice", "Medical procedures",
                    "Emergency medical decisions"
                ],
                "permitted_activities": [
                    "Educational information", "General health guidance",
                    "Symptom documentation", "Healthcare provider referrals"
                ]
            },
            "telemedicine_regulations": {
                "physician_licensing": "Must be licensed in patient's state",
                "prescribing_requirements": "DEA registration required",
                "standard_of_care": "Same as in-person consultation"
            }
        }

        # INTERNATIONAL COMPLIANCE (GDPR, MDR)
        self.international_compliance = {
            "gdpr": {
                "data_protection_principles": [
                    "Lawfulness", "Purpose limitation", "Data minimization",
                    "Accuracy", "Storage limitation", "Security"
                ],
                "individual_rights": [
                    "Right to information", "Right of access",
                    "Right to rectification", "Right to erasure"
                ]
            },
            "eu_mdr": {
                "medical_device_classification": "Risk-based classification",
                "conformity_assessment": "CE marking required",
                "post_market_surveillance": "Vigilance reporting required"
            }
        }

        # PROFESSIONAL LIABILITY INSURANCE REQUIREMENTS
        self.insurance_requirements = {
            "coverage_types": [
                "Professional liability", "General liability",
                "Cyber liability", "Product liability"
            ],
            "minimum_coverage": "$1M per occurrence, $3M aggregate",
            "required_provisions": [
                "AI/ML specific coverage", "Data breach protection",
                "Regulatory defense", "Crisis management"
            ]
        }

        # AUDIT TRAIL REQUIREMENTS
        self.audit_requirements = {
            "required_logs": [
                "User interactions", "AI responses", "Safety interventions",
                "Clinical validations", "System failures"
            ],
            "retention_period": "7 years minimum",
            "access_controls": "Role-based with authentication",
            "integrity_protection": "Tamper-evident logging"
        }

    def assess_legal_compliance(self,
                              user_query: str,
                              ai_response: str,
                              safety_analysis: Dict,
                              clinical_validation: Dict) -> ComplianceRecord:
        """
        Comprehensive legal compliance assessment

        Args:
            user_query: Patient's medical query
            ai_response: AI-generated response
            safety_analysis: Medical safety assessment results
            clinical_validation: Clinical validation results

        Returns:
            ComplianceRecord with legal assessment
        """

        # Generate unique record ID
        record_id = str(uuid.uuid4())

        # Determine compliance level
        compliance_level = self._assess_compliance_level(
            user_query, ai_response, safety_analysis, clinical_validation
        )

        # Assess liability risk
        liability_risk = self._assess_liability_risk(
            user_query, ai_response, safety_analysis, clinical_validation
        )

        # Create privacy-preserving hashes
        user_query_hash = self._create_privacy_hash(user_query)
        ai_response_hash = self._create_privacy_hash(ai_response)

        # Check disclaimer compliance
        disclaimers_present = self._verify_disclaimers(ai_response)

        # Verify emergency protocol compliance
        emergency_protocols_followed = self._verify_emergency_protocols(
            safety_analysis
        )

        # Check professional supervision requirements
        professional_supervision = self._assess_supervision_needs(
            compliance_level, liability_risk
        )

        # Generate audit trail
        audit_trail = self._generate_audit_trail(
            user_query, ai_response, safety_analysis, clinical_validation
        )

        # Generate compliance notes
        compliance_notes = self._generate_compliance_notes(
            compliance_level, liability_risk, disclaimers_present
        )

        # Generate legal recommendations
        legal_recommendations = self._generate_legal_recommendations(
            compliance_level, liability_risk
        )

        return ComplianceRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            compliance_level=compliance_level,
            liability_risk=liability_risk,
            user_query_hash=user_query_hash,
            ai_response_hash=ai_response_hash,
            disclaimers_present=disclaimers_present,
            emergency_protocols_followed=emergency_protocols_followed,
            professional_supervision=professional_supervision,
            audit_trail=audit_trail,
            compliance_notes=compliance_notes,
            legal_recommendations=legal_recommendations
        )

    def _assess_compliance_level(self,
                               user_query: str,
                               ai_response: str,
                               safety_analysis: Dict,
                               clinical_validation: Dict) -> ComplianceLevel:
        """Determine required legal compliance level"""

        # Critical compliance required for emergency situations
        if safety_analysis.get("emergency_detected", False):
            return ComplianceLevel.CRITICAL

        # High compliance for urgent medical conditions
        if safety_analysis.get("urgent_detected", False):
            return ComplianceLevel.HIGH

        # High compliance for clinical recommendations
        if any(keyword in ai_response.lower() for keyword in [
            "medication", "drug", "prescription", "treatment", "therapy"
        ]):
            return ComplianceLevel.HIGH

        # Moderate compliance for general medical advice
        if any(keyword in user_query.lower() for keyword in [
            "symptom", "diagnosis", "medical", "health", "doctor"
        ]):
            return ComplianceLevel.MODERATE

        return ComplianceLevel.LOW

    def _assess_liability_risk(self,
                             user_query: str,
                             ai_response: str,
                             safety_analysis: Dict,
                             clinical_validation: Dict) -> LiabilityRisk:
        """Assess potential liability risk"""

        # Extreme risk for emergency situations without proper disclaimers
        if safety_analysis.get("emergency_detected", False):
            if not self._verify_emergency_disclaimers(ai_response):
                return LiabilityRisk.EXTREME

        # High risk for dangerous advice
        if safety_analysis.get("dangerous_advice_detected", False):
            return LiabilityRisk.HIGH

        # High risk for low clinical accuracy
        clinical_accuracy = clinical_validation.get("clinical_accuracy_score", 1.0)
        if clinical_accuracy < 0.6:
            return LiabilityRisk.HIGH

        # Moderate risk for medical advice without proper disclaimers
        if not self._verify_disclaimers(ai_response):
            return LiabilityRisk.MODERATE

        return LiabilityRisk.LOW

    def _create_privacy_hash(self, text: str) -> str:
        """Create privacy-preserving hash of sensitive text"""
        # Use SHA-256 with salt for privacy protection
        salt = "medico_ai_privacy_salt"
        return hashlib.sha256((text + salt).encode()).hexdigest()[:16]

    def _verify_disclaimers(self, ai_response: str) -> bool:
        """Verify presence of required legal disclaimers"""
        response_lower = ai_response.lower()

        required_elements = [
            "not a substitute for professional medical advice",
            "consult", "healthcare provider"
        ]

        return all(element in response_lower for element in required_elements)

    def _verify_emergency_disclaimers(self, ai_response: str) -> bool:
        """Verify presence of emergency-specific disclaimers"""
        response_lower = ai_response.lower()

        emergency_elements = [
            "emergency", "911", "immediate medical attention"
        ]

        return any(element in response_lower for element in emergency_elements)

    def _verify_emergency_protocols(self, safety_analysis: Dict) -> bool:
        """Verify emergency protocol compliance"""
        if safety_analysis.get("emergency_detected", False):
            return safety_analysis.get("blocked_response", False)
        return True

    def _assess_supervision_needs(self,
                                compliance_level: ComplianceLevel,
                                liability_risk: LiabilityRisk) -> bool:
        """Determine if professional supervision is required"""
        return (compliance_level in [ComplianceLevel.CRITICAL, ComplianceLevel.HIGH] or
                liability_risk in [LiabilityRisk.EXTREME, LiabilityRisk.HIGH])

    def _generate_audit_trail(self,
                            user_query: str,
                            ai_response: str,
                            safety_analysis: Dict,
                            clinical_validation: Dict) -> List[str]:
        """Generate comprehensive audit trail"""
        trail = [
            f"Query received: {datetime.now().isoformat()}",
            f"Safety analysis completed: {safety_analysis.get('safety_level', 'unknown')}",
            f"Clinical validation: {clinical_validation.get('validation_level', 'unknown')}",
            f"Emergency detected: {safety_analysis.get('emergency_detected', False)}",
            f"Response blocked: {safety_analysis.get('blocked_response', False)}"
        ]

        return trail

    def _generate_compliance_notes(self,
                                 compliance_level: ComplianceLevel,
                                 liability_risk: LiabilityRisk,
                                 disclaimers_present: bool) -> List[str]:
        """Generate legal compliance notes"""
        notes = []

        if compliance_level == ComplianceLevel.CRITICAL:
            notes.append("CRITICAL: Full legal review required for this interaction")

        if liability_risk in [LiabilityRisk.EXTREME, LiabilityRisk.HIGH]:
            notes.append("HIGH LIABILITY: Enhanced protections required")

        if not disclaimers_present:
            notes.append("WARNING: Required medical disclaimers missing")

        return notes

    def _generate_legal_recommendations(self,
                                      compliance_level: ComplianceLevel,
                                      liability_risk: LiabilityRisk) -> List[str]:
        """Generate legal recommendations"""
        recommendations = []

        if compliance_level == ComplianceLevel.CRITICAL:
            recommendations.extend([
                "Immediate legal counsel consultation required",
                "Document all safety interventions",
                "Ensure licensed medical professional oversight"
            ])

        if liability_risk == LiabilityRisk.EXTREME:
            recommendations.extend([
                "Block response pending legal review",
                "Implement additional safety measures",
                "Consider system shutdown if pattern continues"
            ])

        recommendations.extend([
            "Maintain comprehensive audit logs",
            "Regular legal compliance review recommended",
            "Update professional liability insurance coverage"
        ])

        return recommendations

    def generate_legal_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive legal compliance summary"""
        return {
            "compliance_framework": {
                "fda_considerations": "Medical device software classification required",
                "hipaa_compliance": "PHI protection and BAA requirements",
                "malpractice_protection": "Professional supervision and disclaimers",
                "state_regulations": "Medical practice law compliance",
                "international": "GDPR and EU MDR considerations"
            },
            "liability_protections": {
                "required_disclaimers": self.malpractice_protections["required_disclaimers"],
                "professional_oversight": "Licensed healthcare provider supervision",
                "insurance_requirements": self.insurance_requirements["minimum_coverage"],
                "audit_trail": "7-year retention with tamper protection"
            },
            "recommendations": [
                "Engage qualified healthcare attorney",
                "Implement comprehensive audit system",
                "Establish clinical oversight protocols",
                "Obtain appropriate liability insurance",
                "Regular compliance reviews and updates"
            ]
        }

# Global legal compliance instance
legal_compliance = LegalComplianceFramework()
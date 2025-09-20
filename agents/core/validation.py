"""
Enterprise Validation Framework for CodeMind Agents
===================================================

Comprehensive validation system with:
- Schema validation using Pydantic
- Business rule validation
- Cross-field validation
- Custom validation rules
- Detailed error reporting
"""

import re
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Type, Union, Callable
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic.fields import FieldInfo

from .exceptions import (
    InputValidationError,
    OutputValidationError,
    BusinessLogicError,
    ConstraintViolationError
)


class ValidationType(str, Enum):
    """Types of validation"""
    SCHEMA = "schema"
    BUSINESS_RULE = "business_rule"
    CONSTRAINT = "constraint"
    CROSS_FIELD = "cross_field"
    CUSTOM = "custom"


class ValidationSeverity(str, Enum):
    """Validation error severity"""
    ERROR = "error"      # Blocks execution
    WARNING = "warning"  # Logs but continues
    INFO = "info"        # Informational only


class ValidationRule(ABC):
    """Base class for validation rules"""

    def __init__(
        self,
        name: str,
        description: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        enabled: bool = True
    ):
        self.name = name
        self.description = description
        self.severity = severity
        self.enabled = enabled

    @abstractmethod
    def validate(self, value: Any, context: Dict[str, Any] = None) -> 'ValidationResult':
        """Execute validation rule"""
        pass


class ValidationResult:
    """Result of a validation operation"""

    def __init__(
        self,
        is_valid: bool,
        rule_name: str,
        severity: ValidationSeverity = ValidationSeverity.ERROR,
        message: str = None,
        details: Dict[str, Any] = None
    ):
        self.is_valid = is_valid
        self.rule_name = rule_name
        self.severity = severity
        self.message = message or ("Validation passed" if is_valid else "Validation failed")
        self.details = details or {}
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


# Input Validation Schemas
class BusinessDescriptionInput(BaseModel):
    """Schema for business description input"""

    business_description: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Natural language description of the business"
    )
    complexity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Complexity score from 0.0 (simple) to 1.0 (complex)"
    )
    user_requirements: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional user requirements"
    )
    constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Business and technical constraints"
    )
    target_timeline: Optional[str] = Field(
        default=None,
        pattern=r'^\d+\s*(days?|weeks?|months?)$',
        description="Target timeline (e.g., '2 weeks', '3 months')"
    )
    budget_range: Optional[str] = Field(
        default=None,
        pattern=r'^\$\d+(\.\d{2})?(\s*-\s*\$\d+(\.\d{2})?)?$',
        description="Budget range (e.g., '$5000', '$5000 - $10000')"
    )

    @field_validator('business_description')
    @classmethod
    def validate_business_description(cls, v):
        """Validate business description content"""
        if not v.strip():
            raise ValueError("Business description cannot be empty")

        # Check for meaningful content
        word_count = len(v.split())
        if word_count < 5:
            raise ValueError("Business description must contain at least 5 words")

        # Check for common spam/test patterns
        spam_patterns = [
            r'^test\s*$',
            r'^hello\s*$',
            r'^[abc]+\s*$',
            r'^\d+\s*$'
        ]

        for pattern in spam_patterns:
            if re.match(pattern, v.strip(), re.IGNORECASE):
                raise ValueError("Business description appears to be test data")

        return v.strip()

    @field_validator('complexity')
    @classmethod
    def validate_complexity(cls, v):
        """Validate complexity score"""
        if not isinstance(v, (int, float)):
            raise ValueError("Complexity must be a number")
        return float(v)


class AgentExecutionInput(BaseModel):
    """Schema for agent execution input"""

    context: Dict[str, Any] = Field(..., description="Shared context")
    agent_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Agent-specific configuration"
    )
    timeout_seconds: Optional[int] = Field(
        default=300,
        ge=1,
        le=3600,
        description="Execution timeout in seconds"
    )
    retry_count: Optional[int] = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retry attempts"
    )

    @field_validator('context')
    @classmethod
    def validate_context(cls, v):
        """Validate context structure"""
        required_fields = ['request_id', 'session_id']

        for field in required_fields:
            if field not in v:
                raise ValueError(f"Context missing required field: {field}")

        return v


# Output Validation Schemas
class PlanningAgentOutput(BaseModel):
    """Schema for planning agent output"""

    business_analysis: Dict[str, Any] = Field(
        ...,
        description="Comprehensive business analysis"
    )
    feature_list: List[str] = Field(
        ...,
        min_length=1,
        description="List of identified features"
    )
    technical_specifications: Dict[str, Any] = Field(
        ...,
        description="Technical specifications and recommendations"
    )
    constraints: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Identified constraints"
    )
    decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Decisions made during planning"
    )
    project_roadmap: Dict[str, Any] = Field(
        ...,
        description="Project development roadmap"
    )
    planning_metadata: Dict[str, Any] = Field(
        ...,
        description="Planning process metadata"
    )

    @field_validator('business_analysis')
    @classmethod
    def validate_business_analysis(cls, v):
        """Validate business analysis structure"""
        required_fields = ['industry', 'business_model', 'confidence']

        for field in required_fields:
            if field not in v:
                raise ValueError(f"Business analysis missing required field: {field}")

        # Validate confidence score
        if not isinstance(v.get('confidence'), (int, float)) or not 0 <= v['confidence'] <= 1:
            raise ValueError("Confidence must be a number between 0 and 1")

        return v

    @field_validator('feature_list')
    @classmethod
    def validate_feature_list(cls, v):
        """Validate feature list"""
        if not v:
            raise ValueError("Feature list cannot be empty")

        # Check for duplicate features
        seen = set()
        duplicates = set()
        for feature in v:
            if feature.lower() in seen:
                duplicates.add(feature)
            seen.add(feature.lower())

        if duplicates:
            raise ValueError(f"Duplicate features found: {duplicates}")

        return v

    @field_validator('project_roadmap')
    @classmethod
    def validate_project_roadmap(cls, v):
        """Validate project roadmap structure"""
        required_fields = ['phases', 'estimated_duration_weeks']

        for field in required_fields:
            if field not in v:
                raise ValueError(f"Project roadmap missing required field: {field}")

        return v


# Custom Validation Rules
class BusinessDescriptionRule(ValidationRule):
    """Validation rule for business descriptions"""

    def __init__(self):
        super().__init__(
            name="business_description_quality",
            description="Validates business description quality and completeness"
        )

    def validate(self, value: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate business description quality"""

        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message="Business description must be a string"
            )

        value = value.strip()

        # Length check
        if len(value) < 20:
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message="Business description too short (minimum 20 characters)"
            )

        # Word count check
        words = value.split()
        if len(words) < 5:
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message="Business description must contain at least 5 words"
            )

        # Check for business-relevant keywords
        business_keywords = [
            'business', 'service', 'product', 'customer', 'user', 'market',
            'solution', 'platform', 'application', 'system', 'tool', 'app'
        ]

        has_business_context = any(
            keyword in value.lower() for keyword in business_keywords
        )

        if not has_business_context:
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                severity=ValidationSeverity.WARNING,
                message="Business description lacks business context keywords"
            )

        return ValidationResult(
            is_valid=True,
            rule_name=self.name,
            message="Business description validation passed"
        )


class IndustryClassificationRule(ValidationRule):
    """Validation rule for industry classification"""

    def __init__(self):
        super().__init__(
            name="industry_classification",
            description="Validates industry classification accuracy"
        )

        self.valid_industries = {
            "E-commerce & Retail", "Healthcare & Medical", "Financial Services",
            "Education & Training", "Real Estate", "Food & Restaurant",
            "Travel & Tourism", "Entertainment & Media", "Business Services",
            "Technology & Software", "Manufacturing", "Agriculture",
            "Transportation & Logistics", "Energy & Utilities", "Government",
            "Non-profit", "Consulting", "Legal Services", "Marketing & Advertising"
        }

    def validate(self, value: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate industry classification"""

        if not isinstance(value, str):
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message="Industry must be a string"
            )

        if value not in self.valid_industries:
            # Check for close matches
            close_matches = [
                industry for industry in self.valid_industries
                if industry.lower() in value.lower() or value.lower() in industry.lower()
            ]

            details = {
                "provided_industry": value,
                "valid_industries": list(self.valid_industries),
                "suggestions": close_matches
            }

            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message=f"Invalid industry classification: {value}",
                details=details
            )

        return ValidationResult(
            is_valid=True,
            rule_name=self.name,
            message="Industry classification is valid"
        )


class ComplexityAssessmentRule(ValidationRule):
    """Validation rule for complexity assessment"""

    def __init__(self):
        super().__init__(
            name="complexity_assessment",
            description="Validates complexity score consistency with business description"
        )

    def validate(self, value: float, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate complexity assessment"""

        if not isinstance(value, (int, float)):
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message="Complexity must be a number"
            )

        if not 0 <= value <= 1:
            return ValidationResult(
                is_valid=False,
                rule_name=self.name,
                message="Complexity must be between 0.0 and 1.0"
            )

        # Check consistency with business description if provided
        if context and 'business_description' in context:
            description = context['business_description'].lower()

            # High complexity indicators
            high_complexity_keywords = [
                'ai', 'machine learning', 'ml', 'artificial intelligence',
                'real-time', 'scalable', 'enterprise', 'complex',
                'advanced', 'sophisticated', 'integration', 'api',
                'microservices', 'distributed', 'blockchain'
            ]

            # Low complexity indicators
            low_complexity_keywords = [
                'simple', 'basic', 'minimal', 'straightforward',
                'landing page', 'brochure', 'static', 'crud'
            ]

            high_indicators = sum(1 for keyword in high_complexity_keywords if keyword in description)
            low_indicators = sum(1 for keyword in low_complexity_keywords if keyword in description)

            # Inconsistency checks
            if value < 0.3 and high_indicators > 2:
                return ValidationResult(
                    is_valid=False,
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message="Complexity score seems low for the described business requirements",
                    details={
                        "complexity_score": value,
                        "high_complexity_indicators": high_indicators,
                        "suggestion": "Consider increasing complexity score"
                    }
                )

            if value > 0.7 and low_indicators > 1 and high_indicators == 0:
                return ValidationResult(
                    is_valid=False,
                    rule_name=self.name,
                    severity=ValidationSeverity.WARNING,
                    message="Complexity score seems high for the described business requirements",
                    details={
                        "complexity_score": value,
                        "low_complexity_indicators": low_indicators,
                        "suggestion": "Consider decreasing complexity score"
                    }
                )

        return ValidationResult(
            is_valid=True,
            rule_name=self.name,
            message="Complexity assessment is valid"
        )


class AgentValidator:
    """
    Comprehensive validator for agent inputs and outputs.

    Provides:
    - Schema validation using Pydantic
    - Custom business rule validation
    - Cross-field validation
    - Detailed error reporting
    """

    def __init__(self):
        self.rules: List[ValidationRule] = []
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default validation rules"""
        self.rules.extend([
            BusinessDescriptionRule(),
            IndustryClassificationRule(),
            ComplexityAssessmentRule()
        ])

    def add_rule(self, rule: ValidationRule):
        """Add custom validation rule"""
        self.rules.append(rule)

    def validate_input(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        agent_id: str = None,
        request_id: str = None
    ) -> Dict[str, Any]:
        """
        Validate agent input against schema and business rules.

        Args:
            data: Input data to validate
            schema: Pydantic schema to validate against
            agent_id: Agent ID for error context
            request_id: Request ID for error context

        Returns:
            Validated and normalized data

        Raises:
            InputValidationError: If validation fails
        """
        try:
            # Schema validation
            validated_data = schema(**data)

            # Convert back to dict for additional validation
            data_dict = validated_data.model_dump()

            # Apply custom validation rules
            validation_errors = []
            validation_warnings = []

            for rule in self.rules:
                if not rule.enabled:
                    continue

                # Determine which field this rule applies to
                field_value = self._extract_field_for_rule(rule, data_dict)
                if field_value is not None:
                    result = rule.validate(field_value, data_dict)

                    if not result.is_valid:
                        if result.severity == ValidationSeverity.ERROR:
                            validation_errors.append(result)
                        elif result.severity == ValidationSeverity.WARNING:
                            validation_warnings.append(result)

            # Raise error if any validation failed
            if validation_errors:
                error_messages = [result.message for result in validation_errors]
                raise InputValidationError(
                    field_name="multiple",
                    field_value=data,
                    validation_errors=error_messages,
                    agent_id=agent_id,
                    request_id=request_id,
                    context={
                        "validation_results": [r.to_dict() for r in validation_errors],
                        "warnings": [r.to_dict() for r in validation_warnings]
                    }
                )

            # Add warnings to result
            if validation_warnings:
                data_dict["_validation_warnings"] = [r.to_dict() for r in validation_warnings]

            return data_dict

        except ValidationError as e:
            # Convert Pydantic validation errors
            error_details = []
            for error in e.errors():
                error_details.append({
                    "field": ".".join(str(loc) for loc in error["loc"]),
                    "message": error["msg"],
                    "type": error["type"]
                })

            raise InputValidationError(
                field_name="schema_validation",
                field_value=data,
                validation_errors=[detail["message"] for detail in error_details],
                agent_id=agent_id,
                request_id=request_id,
                context={"validation_details": error_details}
            )

    def validate_output(
        self,
        data: Dict[str, Any],
        schema: Type[BaseModel],
        agent_id: str = None,
        request_id: str = None
    ) -> Dict[str, Any]:
        """
        Validate agent output against schema.

        Args:
            data: Output data to validate
            schema: Pydantic schema to validate against
            agent_id: Agent ID for error context
            request_id: Request ID for error context

        Returns:
            Validated output data

        Raises:
            OutputValidationError: If validation fails
        """
        try:
            validated_data = schema(**data)
            return validated_data.model_dump()

        except ValidationError as e:
            missing_fields = []
            invalid_fields = []

            for error in e.errors():
                field_name = ".".join(str(loc) for loc in error["loc"])

                if error["type"] == "value_error.missing":
                    missing_fields.append(field_name)
                else:
                    invalid_fields.append(field_name)

            raise OutputValidationError(
                missing_fields=missing_fields,
                invalid_fields=invalid_fields,
                agent_id=agent_id,
                request_id=request_id,
                context={
                    "validation_errors": [
                        {
                            "field": ".".join(str(loc) for loc in error["loc"]),
                            "message": error["msg"],
                            "type": error["type"]
                        }
                        for error in e.errors()
                    ]
                }
            )

    def _extract_field_for_rule(self, rule: ValidationRule, data: Dict[str, Any]) -> Any:
        """Extract the appropriate field value for a validation rule"""

        # Map rules to data fields
        rule_field_mapping = {
            "business_description_quality": "business_description",
            "industry_classification": ("business_analysis", "industry"),
            "complexity_assessment": "complexity"
        }

        field_path = rule_field_mapping.get(rule.name)
        if not field_path:
            return None

        if isinstance(field_path, str):
            return data.get(field_path)
        elif isinstance(field_path, tuple):
            # Nested field access
            current = data
            for key in field_path:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return None
            return current

        return None


# Global validator instance
agent_validator = AgentValidator()


def get_validator() -> AgentValidator:
    """Get the global agent validator instance"""
    return agent_validator
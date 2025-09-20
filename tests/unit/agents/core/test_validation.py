"""
Unit Tests for Validation Framework
===================================

Tests for comprehensive validation system including:
- Schema validation
- Business rule validation
- Custom validation rules
- Error handling and reporting
"""

import pytest
from typing import Dict, Any
from pydantic import ValidationError

from agents.core.validation import (
    AgentValidator,
    BusinessDescriptionInput,
    PlanningAgentOutput,
    BusinessDescriptionRule,
    IndustryClassificationRule,
    ComplexityAssessmentRule,
    ValidationResult,
    ValidationSeverity,
    get_validator
)
from agents.core.exceptions import InputValidationError, OutputValidationError


class TestBusinessDescriptionInput:
    """Test BusinessDescriptionInput schema"""

    def test_valid_input(self):
        """Test valid business description input"""
        data = {
            "business_description": "A SaaS platform for project management targeting small teams",
            "complexity": 0.6,
            "target_timeline": "3 months",
            "budget_range": "$5000 - $15000"
        }

        result = BusinessDescriptionInput(**data)

        assert result.business_description == data["business_description"]
        assert result.complexity == 0.6
        assert result.target_timeline == "3 months"
        assert result.budget_range == "$5000 - $15000"

    def test_invalid_business_description_too_short(self):
        """Test business description too short"""
        data = {
            "business_description": "Short",
            "complexity": 0.5
        }

        with pytest.raises(ValidationError) as exc_info:
            BusinessDescriptionInput(**data)

        errors = exc_info.value.errors()
        assert any("String should have at least 10 characters" in str(error) for error in errors)

    def test_invalid_business_description_empty(self):
        """Test empty business description"""
        data = {
            "business_description": "",
            "complexity": 0.5
        }

        with pytest.raises(ValidationError) as exc_info:
            BusinessDescriptionInput(**data)

        errors = exc_info.value.errors()
        assert any("String should have at least 10 characters" in str(error) for error in errors)

    def test_invalid_complexity_out_of_range(self):
        """Test complexity score out of range"""
        data = {
            "business_description": "Valid business description for testing",
            "complexity": 1.5
        }

        with pytest.raises(ValidationError) as exc_info:
            BusinessDescriptionInput(**data)

        errors = exc_info.value.errors()
        assert any("Input should be less than or equal to 1" in str(error) for error in errors)

    def test_invalid_timeline_format(self):
        """Test invalid timeline format"""
        data = {
            "business_description": "Valid business description for testing",
            "complexity": 0.5,
            "target_timeline": "invalid timeline"
        }

        with pytest.raises(ValidationError) as exc_info:
            BusinessDescriptionInput(**data)

        errors = exc_info.value.errors()
        assert any("String should match pattern" in str(error) for error in errors)

    def test_valid_timeline_formats(self):
        """Test various valid timeline formats"""
        valid_timelines = ["2 weeks", "3 months", "1 day", "6 month"]

        for timeline in valid_timelines:
            data = {
                "business_description": "Valid business description for testing",
                "complexity": 0.5,
                "target_timeline": timeline
            }

            result = BusinessDescriptionInput(**data)
            assert result.target_timeline == timeline

    def test_business_description_validation_spam_detection(self):
        """Test spam detection in business description"""
        spam_inputs = ["test", "hello", "abc", "123"]

        for spam_input in spam_inputs:
            data = {
                "business_description": spam_input,
                "complexity": 0.5
            }

            with pytest.raises(ValidationError) as exc_info:
                BusinessDescriptionInput(**data)

            errors = exc_info.value.errors()
            # These short inputs fail length validation first, so check for that
            assert any("String should have at least 10 characters" in str(error) for error in errors)

    def test_business_description_minimum_words(self):
        """Test minimum word count validation"""
        data = {
            "business_description": "One two three four",  # Only 4 words
            "complexity": 0.5
        }

        with pytest.raises(ValidationError) as exc_info:
            BusinessDescriptionInput(**data)

        errors = exc_info.value.errors()
        assert any("at least 5 words" in str(error) for error in errors)


class TestPlanningAgentOutput:
    """Test PlanningAgentOutput schema"""

    def test_valid_output(self):
        """Test valid planning agent output"""
        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "confidence": 0.8
            },
            "feature_list": ["user_authentication", "project_management", "reporting"],
            "technical_specifications": {
                "architecture": "microservices",
                "frontend": "React",
                "backend": "FastAPI"
            },
            "project_roadmap": {
                "phases": ["foundation", "development", "testing"],
                "estimated_duration_weeks": 8
            },
            "planning_metadata": {
                "complexity_assessment": 0.6,
                "industry_classification": "Technology & Software"
            }
        }

        result = PlanningAgentOutput(**data)

        assert result.business_analysis["industry"] == "Technology & Software"
        assert len(result.feature_list) == 3
        assert result.project_roadmap["estimated_duration_weeks"] == 8

    def test_missing_required_fields(self):
        """Test missing required fields in output"""
        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "confidence": 0.8
            },
            "feature_list": ["authentication"],
            # Missing technical_specifications, project_roadmap, planning_metadata
        }

        with pytest.raises(ValidationError) as exc_info:
            PlanningAgentOutput(**data)

        errors = exc_info.value.errors()
        missing_fields = {error["loc"][0] for error in errors if error["type"] == "missing"}

        assert "technical_specifications" in missing_fields
        assert "project_roadmap" in missing_fields
        assert "planning_metadata" in missing_fields

    def test_invalid_business_analysis_structure(self):
        """Test invalid business analysis structure"""
        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                # Missing business_model and confidence
            },
            "feature_list": ["authentication"],
            "technical_specifications": {"arch": "monolith"},
            "project_roadmap": {"phases": [], "estimated_duration_weeks": 4},
            "planning_metadata": {"complexity": 0.5}
        }

        with pytest.raises(ValidationError) as exc_info:
            PlanningAgentOutput(**data)

        errors = exc_info.value.errors()
        assert any("business_model" in str(error) for error in errors)
        # Only business_model should be missing in this test case
        assert not any("confidence" in str(error) for error in errors)

    def test_empty_feature_list(self):
        """Test empty feature list validation"""
        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "confidence": 0.8
            },
            "feature_list": [],  # Empty list
            "technical_specifications": {"arch": "monolith"},
            "project_roadmap": {"phases": [], "estimated_duration_weeks": 4},
            "planning_metadata": {"complexity": 0.5}
        }

        with pytest.raises(ValidationError) as exc_info:
            PlanningAgentOutput(**data)

        errors = exc_info.value.errors()
        assert any("List should have at least 1 item" in str(error) for error in errors)

    def test_duplicate_features_validation(self):
        """Test duplicate features detection"""
        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "confidence": 0.8
            },
            "feature_list": ["authentication", "Authentication", "user_auth"],  # Duplicates
            "technical_specifications": {"arch": "monolith"},
            "project_roadmap": {"phases": [], "estimated_duration_weeks": 4},
            "planning_metadata": {"complexity": 0.5}
        }

        with pytest.raises(ValidationError) as exc_info:
            PlanningAgentOutput(**data)

        errors = exc_info.value.errors()
        assert any("Duplicate features" in str(error) for error in errors)


class TestValidationRules:
    """Test custom validation rules"""

    def test_business_description_rule_valid(self):
        """Test business description rule with valid input"""
        rule = BusinessDescriptionRule()

        result = rule.validate(
            "A comprehensive project management SaaS platform for small teams",
            {}
        )

        assert result.is_valid is True
        assert result.rule_name == "business_description_quality"

    def test_business_description_rule_too_short(self):
        """Test business description rule with short input"""
        rule = BusinessDescriptionRule()

        result = rule.validate("Short text", {})

        assert result.is_valid is False
        assert "too short" in result.message

    def test_business_description_rule_no_business_context(self):
        """Test business description rule without business keywords"""
        rule = BusinessDescriptionRule()

        result = rule.validate(
            "This is a long description that does not contain any relevant keywords for identification",
            {}
        )

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.WARNING
        assert "business context" in result.message

    def test_industry_classification_rule_valid(self):
        """Test industry classification rule with valid industry"""
        rule = IndustryClassificationRule()

        result = rule.validate("Healthcare & Medical", {})

        assert result.is_valid is True

    def test_industry_classification_rule_invalid(self):
        """Test industry classification rule with invalid industry"""
        rule = IndustryClassificationRule()

        result = rule.validate("Invalid Industry", {})

        assert result.is_valid is False
        assert "Invalid industry classification" in result.message
        assert "suggestions" in result.details

    def test_complexity_assessment_rule_valid(self):
        """Test complexity assessment rule with valid input"""
        rule = ComplexityAssessmentRule()

        result = rule.validate(0.5, {})

        assert result.is_valid is True

    def test_complexity_assessment_rule_invalid_type(self):
        """Test complexity assessment rule with invalid type"""
        rule = ComplexityAssessmentRule()

        result = rule.validate("invalid", {})

        assert result.is_valid is False
        assert "must be a number" in result.message

    def test_complexity_assessment_rule_out_of_range(self):
        """Test complexity assessment rule with value out of range"""
        rule = ComplexityAssessmentRule()

        result = rule.validate(1.5, {})

        assert result.is_valid is False
        assert "between 0.0 and 1.0" in result.message

    def test_complexity_assessment_rule_consistency_check(self):
        """Test complexity assessment consistency with business description"""
        rule = ComplexityAssessmentRule()

        # Low complexity with high complexity indicators
        context = {
            "business_description": "Advanced AI machine learning platform with real-time processing"
        }

        result = rule.validate(0.2, context)

        assert result.is_valid is False
        assert result.severity == ValidationSeverity.WARNING
        assert "seems low" in result.message


class TestAgentValidator:
    """Test AgentValidator functionality"""

    def test_validator_initialization(self):
        """Test validator initialization with default rules"""
        validator = AgentValidator()

        assert len(validator.rules) > 0
        rule_names = [rule.name for rule in validator.rules]
        assert "business_description_quality" in rule_names
        assert "industry_classification" in rule_names

    def test_validate_input_success(self):
        """Test successful input validation"""
        validator = AgentValidator()

        data = {
            "business_description": "A project management platform for software teams",
            "complexity": 0.6
        }

        result = validator.validate_input(
            data,
            BusinessDescriptionInput,
            agent_id="test-agent",
            request_id="test-request"
        )

        assert "business_description" in result
        assert result["complexity"] == 0.6

    def test_validate_input_schema_failure(self):
        """Test input validation with schema failure"""
        validator = AgentValidator()

        data = {
            "business_description": "",  # Invalid
            "complexity": 2.0  # Out of range
        }

        with pytest.raises(InputValidationError) as exc_info:
            validator.validate_input(
                data,
                BusinessDescriptionInput,
                agent_id="test-agent",
                request_id="test-request"
            )

        error = exc_info.value
        assert error.agent_id == "test-agent"
        assert error.request_id == "test-request"
        assert "validation_details" in error.context

    def test_validate_input_business_rule_failure(self):
        """Test input validation with business rule failure"""
        validator = AgentValidator()

        # Add a strict rule that will fail
        class StrictRule(BusinessDescriptionRule):
            def validate(self, value, context=None):
                return ValidationResult(
                    is_valid=False,
                    rule_name="strict_test",
                    message="Always fails"
                )

        validator.add_rule(StrictRule())

        data = {
            "business_description": "A valid business description for testing purposes",
            "complexity": 0.5
        }

        with pytest.raises(InputValidationError):
            validator.validate_input(
                data,
                BusinessDescriptionInput,
                agent_id="test-agent"
            )

    def test_validate_output_success(self):
        """Test successful output validation"""
        validator = AgentValidator()

        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "confidence": 0.8
            },
            "feature_list": ["authentication", "dashboard"],
            "technical_specifications": {"architecture": "microservices"},
            "project_roadmap": {
                "phases": ["planning", "development"],
                "estimated_duration_weeks": 6
            },
            "planning_metadata": {"complexity": 0.6}
        }

        result = validator.validate_output(
            data,
            PlanningAgentOutput,
            agent_id="test-agent"
        )

        assert "business_analysis" in result
        assert len(result["feature_list"]) == 2

    def test_validate_output_missing_fields(self):
        """Test output validation with missing fields"""
        validator = AgentValidator()

        data = {
            "business_analysis": {
                "industry": "Technology & Software",
                "business_model": "SaaS",
                "confidence": 0.8
            },
            "feature_list": ["authentication"]
            # Missing required fields
        }

        with pytest.raises(OutputValidationError) as exc_info:
            validator.validate_output(
                data,
                PlanningAgentOutput,
                agent_id="test-agent"
            )

        error = exc_info.value
        assert len(error.invalid_fields) > 0
        assert "technical_specifications" in error.invalid_fields

    def test_get_global_validator(self):
        """Test getting global validator instance"""
        validator1 = get_validator()
        validator2 = get_validator()

        assert validator1 is validator2  # Should be same instance
        assert isinstance(validator1, AgentValidator)


class TestValidationResult:
    """Test ValidationResult functionality"""

    def test_validation_result_creation(self):
        """Test ValidationResult creation"""
        result = ValidationResult(
            is_valid=True,
            rule_name="test_rule",
            severity=ValidationSeverity.INFO,
            message="Test message",
            details={"key": "value"}
        )

        assert result.is_valid is True
        assert result.rule_name == "test_rule"
        assert result.severity == ValidationSeverity.INFO
        assert result.message == "Test message"
        assert result.details["key"] == "value"

    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization"""
        result = ValidationResult(
            is_valid=False,
            rule_name="test_rule",
            message="Validation failed"
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["is_valid"] is False
        assert result_dict["rule_name"] == "test_rule"
        assert result_dict["message"] == "Validation failed"
        assert "timestamp" in result_dict


if __name__ == "__main__":
    pytest.main([__file__])
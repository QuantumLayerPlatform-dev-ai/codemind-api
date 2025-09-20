"""
Enterprise-Grade Exception Handling for CodeMind Agents
======================================================

Comprehensive exception hierarchy with error codes, recovery strategies,
and detailed logging for production debugging and monitoring.
"""

import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting"""
    CRITICAL = "critical"      # System cannot function
    HIGH = "high"             # Major functionality impacted
    MEDIUM = "medium"         # Some functionality impacted
    LOW = "low"               # Minor issues, system functional
    INFO = "info"             # Informational, no impact


class ErrorCategory(str, Enum):
    """Error categories for classification and handling"""
    VALIDATION = "validation"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    NETWORK = "network"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    BUSINESS_LOGIC = "business_logic"
    EXTERNAL_SERVICE = "external_service"
    DATA_INTEGRITY = "data_integrity"
    SYSTEM = "system"


class RecoveryStrategy(str, Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    MANUAL_INTERVENTION = "manual_intervention"
    IGNORE = "ignore"
    ESCALATE = "escalate"


class AgentErrorBase(Exception):
    """
    Base exception class for all agent errors with enterprise features.

    Provides:
    - Structured error information
    - Error categorization and severity
    - Recovery strategy suggestions
    - Detailed context capture
    - Monitoring integration
    """

    def __init__(
        self,
        message: str,
        error_code: str = None,
        agent_id: str = None,
        request_id: str = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.MANUAL_INTERVENTION,
        context: Dict[str, Any] = None,
        cause: Exception = None,
        user_message: str = None,
        retry_after: int = None,
        recoverable: bool = True
    ):
        super().__init__(message)

        self.message = message
        self.agent_id = agent_id
        self.request_id = request_id
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        self.context = context or {}
        self.cause = cause
        self.user_message = user_message or "An error occurred while processing your request"
        self.retry_after = retry_after
        self.recoverable = recoverable
        self.error_code = error_code or self._generate_error_code()

        # Metadata
        self.timestamp = datetime.now(timezone.utc)
        self.traceback_str = traceback.format_exc()
        self.error_id = self._generate_error_id()

    def _generate_error_code(self) -> str:
        """Generate error code based on exception class"""
        class_name = self.__class__.__name__
        return f"{self.category.value.upper()}_{class_name.upper()}"

    def _generate_error_id(self) -> str:
        """Generate unique error ID for tracking"""
        import uuid
        return str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/monitoring"""
        return {
            "error_id": self.error_id,
            "error_code": self.error_code,
            "message": self.message,
            "user_message": self.user_message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_strategy": self.recovery_strategy.value,
            "agent_id": self.agent_id,
            "request_id": self.request_id,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str
        }

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message} (Agent: {self.agent_id}, Request: {self.request_id})"


# Validation Errors
class AgentValidationError(AgentErrorBase):
    """Raised when agent input or output validation fails"""

    def __init__(self, message: str, validation_errors: List[str] = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            recoverable=False,
            **kwargs
        )
        self.validation_errors = validation_errors or []
        self.context["validation_errors"] = self.validation_errors


class InputValidationError(AgentValidationError):
    """Raised when agent input validation fails"""

    def __init__(self, field_name: str, field_value: Any, expected_type: str = None, **kwargs):
        message = f"Invalid input for field '{field_name}': {field_value}"
        if expected_type:
            message += f" (expected: {expected_type})"

        super().__init__(
            message=message,
            user_message=f"Invalid input provided for {field_name}",
            **kwargs
        )
        self.field_name = field_name
        self.field_value = field_value
        self.expected_type = expected_type


class OutputValidationError(AgentValidationError):
    """Raised when agent output validation fails"""

    def __init__(self, missing_fields: List[str] = None, invalid_fields: List[str] = None, **kwargs):
        message = "Agent output validation failed"
        if missing_fields:
            message += f" - Missing fields: {missing_fields}"
        if invalid_fields:
            message += f" - Invalid fields: {invalid_fields}"

        super().__init__(
            message=message,
            user_message="The system produced invalid results. Please try again.",
            **kwargs
        )
        self.missing_fields = missing_fields or []
        self.invalid_fields = invalid_fields or []


# Execution Errors
class AgentExecutionError(AgentErrorBase):
    """Raised when agent execution fails"""

    def __init__(self, message: str, execution_phase: str = None, **kwargs):
        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.BUSINESS_LOGIC,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs
        )
        self.execution_phase = execution_phase
        if execution_phase:
            self.context["execution_phase"] = execution_phase


class AgentTimeoutError(AgentErrorBase):
    """Raised when agent execution times out"""

    def __init__(self, timeout_seconds: float, operation: str = None, **kwargs):
        message = f"Agent operation timed out after {timeout_seconds} seconds"
        if operation:
            message += f" during {operation}"

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.TIMEOUT,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_after=int(timeout_seconds * 1.5),  # Suggest longer timeout
            user_message="The operation is taking longer than expected. Please try again.",
            **kwargs
        )
        self.timeout_seconds = timeout_seconds
        self.operation = operation


class AgentResourceError(AgentErrorBase):
    """Raised when agent cannot access required resources"""

    def __init__(self, resource_type: str, resource_id: str = None, **kwargs):
        message = f"Cannot access required resource: {resource_type}"
        if resource_id:
            message += f" (ID: {resource_id})"

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.RESOURCE,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_after=30,
            user_message="Required system resources are temporarily unavailable. Please try again shortly.",
            **kwargs
        )
        self.resource_type = resource_type
        self.resource_id = resource_id


# Configuration Errors
class AgentConfigurationError(AgentErrorBase):
    """Raised when agent configuration is invalid"""

    def __init__(self, config_key: str, config_value: Any = None, **kwargs):
        message = f"Invalid configuration for '{config_key}'"
        if config_value is not None:
            message += f": {config_value}"

        super().__init__(
            message=message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.CONFIGURATION,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            recoverable=False,
            user_message="System configuration error. Please contact support.",
            **kwargs
        )
        self.config_key = config_key
        self.config_value = config_value


# External Service Errors
class ExternalServiceError(AgentErrorBase):
    """Raised when external service calls fail"""

    def __init__(self, service_name: str, status_code: int = None, service_message: str = None, **kwargs):
        message = f"External service '{service_name}' failed"
        if status_code:
            message += f" with status {status_code}"
        if service_message:
            message += f": {service_message}"

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_SERVICE,
            recovery_strategy=RecoveryStrategy.CIRCUIT_BREAKER,
            retry_after=60,
            user_message="External service temporarily unavailable. Please try again later.",
            **kwargs
        )
        self.service_name = service_name
        self.status_code = status_code
        self.service_message = service_message


class LLMServiceError(ExternalServiceError):
    """Raised when LLM service calls fail"""

    def __init__(self, model_name: str, error_type: str = None, **kwargs):
        super().__init__(
            service_name=f"LLM Service ({model_name})",
            user_message="AI service temporarily unavailable. Please try again.",
            **kwargs
        )
        self.model_name = model_name
        self.error_type = error_type


# Context and State Errors
class ContextError(AgentErrorBase):
    """Raised when context operations fail"""

    def __init__(self, operation: str, context_id: str = None, **kwargs):
        message = f"Context operation '{operation}' failed"
        if context_id:
            message += f" for context {context_id}"

        super().__init__(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.DATA_INTEGRITY,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_after=10,
            **kwargs
        )
        self.operation = operation
        self.context_id = context_id


class StateCorruptionError(AgentErrorBase):
    """Raised when agent state is corrupted"""

    def __init__(self, state_description: str, **kwargs):
        super().__init__(
            message=f"Agent state corruption detected: {state_description}",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.DATA_INTEGRITY,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            recoverable=False,
            user_message="System state error detected. Your session will be reset.",
            **kwargs
        )
        self.state_description = state_description


# Network and Connectivity Errors
class NetworkError(AgentErrorBase):
    """Raised when network operations fail"""

    def __init__(self, operation: str, endpoint: str = None, **kwargs):
        message = f"Network operation '{operation}' failed"
        if endpoint:
            message += f" for endpoint {endpoint}"

        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.NETWORK,
            recovery_strategy=RecoveryStrategy.RETRY,
            retry_after=30,
            user_message="Network connectivity issue. Please check your connection and try again.",
            **kwargs
        )
        self.operation = operation
        self.endpoint = endpoint


# Business Logic Errors
class BusinessLogicError(AgentErrorBase):
    """Raised when business logic validation fails"""

    def __init__(self, rule_name: str, rule_description: str = None, **kwargs):
        message = f"Business rule violation: {rule_name}"
        if rule_description:
            message += f" - {rule_description}"

        super().__init__(
            message=message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.BUSINESS_LOGIC,
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION,
            recoverable=False,
            user_message="Your request violates business rules. Please review and try again.",
            **kwargs
        )
        self.rule_name = rule_name
        self.rule_description = rule_description


class ConstraintViolationError(BusinessLogicError):
    """Raised when constraints are violated"""

    def __init__(self, constraint_name: str, constraint_value: Any, actual_value: Any, **kwargs):
        super().__init__(
            rule_name=f"Constraint: {constraint_name}",
            rule_description=f"Expected {constraint_value}, got {actual_value}",
            **kwargs
        )
        self.constraint_name = constraint_name
        self.constraint_value = constraint_value
        self.actual_value = actual_value


# Utility Functions
def create_error_response(error: AgentErrorBase) -> Dict[str, Any]:
    """Create standardized error response for APIs"""
    return {
        "success": False,
        "error": {
            "code": error.error_code,
            "message": error.user_message,
            "details": error.message if error.severity in [ErrorSeverity.LOW, ErrorSeverity.INFO] else None,
            "error_id": error.error_id,
            "recoverable": error.recoverable,
            "retry_after": error.retry_after,
            "category": error.category.value
        },
        "timestamp": error.timestamp.isoformat(),
        "request_id": error.request_id
    }


def wrap_external_exception(
    exc: Exception,
    agent_id: str = None,
    request_id: str = None,
    context: Dict[str, Any] = None
) -> AgentErrorBase:
    """Wrap external exceptions in AgentErrorBase"""

    # Map common exceptions to appropriate agent errors
    if isinstance(exc, TimeoutError):
        return AgentTimeoutError(
            timeout_seconds=30,  # Default
            agent_id=agent_id,
            request_id=request_id,
            context=context,
            cause=exc
        )
    elif isinstance(exc, ConnectionError):
        return NetworkError(
            operation="connection",
            agent_id=agent_id,
            request_id=request_id,
            context=context,
            cause=exc
        )
    elif isinstance(exc, ValueError):
        return InputValidationError(
            field_name="unknown",
            field_value=str(exc),
            agent_id=agent_id,
            request_id=request_id,
            context=context,
            cause=exc
        )
    else:
        return AgentExecutionError(
            message=f"Unexpected error: {str(exc)}",
            agent_id=agent_id,
            request_id=request_id,
            context=context,
            cause=exc
        )
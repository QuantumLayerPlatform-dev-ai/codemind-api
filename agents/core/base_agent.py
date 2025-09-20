"""
Base Agent Class for CodeMind Cognitive Software Factory
=======================================================

Provides the foundation for all agents in the system with:
- Context management and sharing
- Agent fingerprinting and traceability
- Observability and metrics
- Error handling and recovery
- Communication protocols
"""

import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from datetime import datetime, timezone
import asyncio

from opentelemetry import trace
from opentelemetry.trace import Span
import psutil

try:
    from ...core.logging import get_logger
    from ...core.config import get_settings
    from ...services.llm_router import LLMRouter
except ImportError:
    from core.logging import get_logger
    from core.config import get_settings
    from services.llm_router import LLMRouter

from .context_manager import SharedContext, ContextManager, Decision, Constraint, get_context_manager
from .fingerprinting import AgentFingerprint, FingerprintGenerator, AgentType, AgentStatus, fingerprint_tracker

logger = get_logger("base_agent")
settings = get_settings()
tracer = trace.get_tracer(__name__)


class AgentError(Exception):
    """Base exception for agent errors"""
    def __init__(self, message: str, agent_id: str, error_code: str = None):
        super().__init__(message)
        self.agent_id = agent_id
        self.error_code = error_code or "AGENT_ERROR"
        self.timestamp = datetime.now(timezone.utc)


class AgentTimeoutError(AgentError):
    """Exception raised when agent execution times out"""
    def __init__(self, agent_id: str, timeout: float):
        super().__init__(f"Agent {agent_id} timed out after {timeout} seconds", agent_id, "TIMEOUT")
        self.timeout = timeout


class AgentValidationError(AgentError):
    """Exception raised when agent input/output validation fails"""
    def __init__(self, agent_id: str, validation_message: str):
        super().__init__(f"Validation failed: {validation_message}", agent_id, "VALIDATION_ERROR")


class BaseAgent(ABC):
    """
    Base class for all CodeMind agents.

    Provides:
    - Context management and state sharing
    - Agent fingerprinting and identity
    - Observability and tracing
    - Error handling and recovery
    - Input/output validation
    - Performance monitoring
    """

    def __init__(
        self,
        context: SharedContext,
        agent_type: AgentType,
        version: str = "1.0.0",
        timeout: float = 300.0,  # 5 minutes default
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize base agent.

        Args:
            context: Shared context for agent collaboration
            agent_type: Type of agent (planning, architecture, code, etc.)
            version: Agent version
            timeout: Maximum execution time in seconds
            max_retries: Maximum number of retry attempts
            **kwargs: Additional agent-specific parameters
        """
        self.context = context
        self.timeout = timeout
        self.max_retries = max_retries

        # Create agent fingerprint
        self.fingerprint = FingerprintGenerator.create_fingerprint(
            agent_type=agent_type,
            version=version,
            model_config=kwargs.get('model_config', {}),
            parameters=kwargs,
            parent_agent_id=kwargs.get('parent_agent_id')
        )

        # Register with tracker
        fingerprint_tracker.register_fingerprint(self.fingerprint)

        # Initialize managers
        self.context_manager: Optional[ContextManager] = None
        self.llm_router: Optional[LLMRouter] = None

        # State tracking
        self._start_time: Optional[float] = None
        self._span: Optional[Span] = None

        logger.info(f"Initialized {agent_type} agent: {self.fingerprint.agent_id}")

    async def initialize(self) -> None:
        """Initialize agent dependencies"""
        try:
            # Initialize context manager
            self.context_manager = await get_context_manager()

            # Initialize LLM router
            self.llm_router = LLMRouter()

            # Update fingerprint status
            self.fingerprint.update_status(AgentStatus.ACTIVE)

            logger.debug(f"Agent {self.fingerprint.agent_id} initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize agent {self.fingerprint.agent_id}: {e}")
            self.fingerprint.update_status(AgentStatus.FAILED, {"error": str(e)})
            raise AgentError(f"Initialization failed: {e}", self.fingerprint.agent_id, "INIT_ERROR")

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent with full observability and error handling.

        Returns:
            Dict containing execution results and metadata
        """
        trace_id = kwargs.get('trace_id', self.context.request_id)

        with tracer.start_as_current_span(f"{self.fingerprint.agent_type}_execution") as span:
            self._span = span
            span.set_attribute("agent.id", self.fingerprint.agent_id)
            span.set_attribute("agent.type", self.fingerprint.agent_type)
            span.set_attribute("request.id", self.context.request_id)

            try:
                # Start execution tracking
                await self._start_execution(trace_id, span.get_span_context().trace_id)

                # Pre-execution validation
                await self._validate_inputs(**kwargs)

                # Create checkpoint
                checkpoint = self.context.create_checkpoint(
                    self.fingerprint.agent_id,
                    f"Pre-execution checkpoint for {self.fingerprint.agent_type}"
                )

                # Execute with timeout
                result = await asyncio.wait_for(
                    self._execute_with_retry(**kwargs),
                    timeout=self.timeout
                )

                # Post-execution validation
                await self._validate_outputs(result)

                # Update context with results
                await self._update_context(result)

                # End execution tracking
                await self._end_execution(success=True)

                span.set_attribute("execution.success", True)
                span.set_attribute("execution.result_size", len(str(result)))

                logger.info(f"Agent {self.fingerprint.agent_id} executed successfully")
                return result

            except asyncio.TimeoutError:
                error = AgentTimeoutError(self.fingerprint.agent_id, self.timeout)
                await self._handle_error(error)
                span.record_exception(error)
                span.set_attribute("execution.success", False)
                raise error

            except Exception as e:
                error = AgentError(f"Execution failed: {e}", self.fingerprint.agent_id)
                await self._handle_error(error)
                span.record_exception(e)
                span.set_attribute("execution.success", False)
                raise error

    async def _execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """Execute with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt} for agent {self.fingerprint.agent_id}")
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)

                # Execute the actual agent logic
                result = await self._execute_impl(**kwargs)
                return result

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed for agent {self.fingerprint.agent_id}: {e}")

                # Don't retry on validation errors
                if isinstance(e, AgentValidationError):
                    break

        # All retries exhausted
        raise last_exception

    @abstractmethod
    async def _execute_impl(self, **kwargs) -> Dict[str, Any]:
        """
        Implement the core agent logic.

        This method must be implemented by each specific agent type.

        Returns:
            Dict containing the agent's output
        """
        pass

    async def _validate_inputs(self, **kwargs) -> None:
        """Validate agent inputs"""
        # Default implementation - can be overridden
        required_fields = getattr(self, 'REQUIRED_INPUTS', [])

        for field in required_fields:
            if field not in kwargs and not hasattr(self.context, field):
                raise AgentValidationError(
                    self.fingerprint.agent_id,
                    f"Required input field '{field}' is missing"
                )

    async def _validate_outputs(self, result: Dict[str, Any]) -> None:
        """Validate agent outputs"""
        # Default implementation - can be overridden
        if not isinstance(result, dict):
            raise AgentValidationError(
                self.fingerprint.agent_id,
                f"Output must be a dictionary, got {type(result)}"
            )

        required_outputs = getattr(self, 'REQUIRED_OUTPUTS', [])
        for field in required_outputs:
            if field not in result:
                raise AgentValidationError(
                    self.fingerprint.agent_id,
                    f"Required output field '{field}' is missing"
                )

    async def _start_execution(self, trace_id: str, span_id: str) -> None:
        """Start execution tracking"""
        self._start_time = time.time()

        # Update fingerprint
        self.fingerprint.start_execution(
            request_id=self.context.request_id,
            trace_id=trace_id,
            span_id=span_id
        )

        # Update context
        if self.fingerprint.agent_id not in self.context.active_agents:
            self.context.active_agents.append(self.fingerprint.agent_id)

        # Update context in storage
        if self.context_manager:
            await self.context_manager.update_context(self.context)

    async def _end_execution(self, success: bool = True, cost: float = 0.0, tokens: int = 0) -> None:
        """End execution tracking"""
        execution_time = time.time() - (self._start_time or time.time())

        # Update fingerprint
        self.fingerprint.end_execution(
            success=success,
            execution_time=execution_time,
            cost=cost,
            tokens=tokens
        )

        # Update context
        if self.fingerprint.agent_id in self.context.active_agents:
            self.context.active_agents.remove(self.fingerprint.agent_id)

        if success:
            if self.fingerprint.agent_id not in self.context.completed_agents:
                self.context.completed_agents.append(self.fingerprint.agent_id)
        else:
            if self.fingerprint.agent_id not in self.context.failed_agents:
                self.context.failed_agents.append(self.fingerprint.agent_id)

        # Collect resource usage
        process = psutil.Process()
        memory_info = process.memory_info()

        self.fingerprint.performance_metrics.memory_usage = {
            "rss": memory_info.rss,
            "vms": memory_info.vms
        }

        self.fingerprint.performance_metrics.cpu_usage = {
            "percent": process.cpu_percent()
        }

        # Update context in storage
        if self.context_manager:
            await self.context_manager.update_context(self.context)

    async def _update_context(self, result: Dict[str, Any]) -> None:
        """Update shared context with agent results"""
        # Add agent outputs to context artifacts
        agent_output_key = f"{self.fingerprint.agent_type}_output"
        self.context.update_artifact(
            agent_output_key,
            result,
            self.fingerprint.agent_id
        )

        # Add any decisions made by this agent
        if 'decisions' in result:
            for decision_data in result['decisions']:
                decision = Decision(
                    agent_id=self.fingerprint.agent_id,
                    agent_type=self.fingerprint.agent_type,
                    **decision_data
                )
                self.context.add_decision(decision)

        # Add any constraints identified by this agent
        if 'constraints' in result:
            for constraint_data in result['constraints']:
                constraint = Constraint(**constraint_data)
                self.context.add_constraint(constraint)

    async def _handle_error(self, error: Exception) -> None:
        """Handle agent errors"""
        await self._end_execution(success=False)

        # Add error information to context
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "agent_id": self.fingerprint.agent_id,
            "agent_type": self.fingerprint.agent_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "traceback": traceback.format_exc()
        }

        self.context.update_artifact(
            f"error_{self.fingerprint.agent_id}",
            error_info,
            self.fingerprint.agent_id
        )

        logger.error(f"Agent {self.fingerprint.agent_id} failed: {error}")

    async def send_message(self, message: Dict[str, Any], target_agent_id: str = None) -> None:
        """Send message to other agents (placeholder for NATS integration)"""
        # TODO: Implement NATS messaging
        logger.info(f"Agent {self.fingerprint.agent_id} sending message: {message}")

    async def cleanup(self) -> None:
        """Cleanup agent resources"""
        try:
            # Unregister from tracker
            fingerprint_tracker.unregister_fingerprint(self.fingerprint.agent_id)

            # Close connections
            if self.context_manager:
                await self.context_manager.close()

            logger.info(f"Agent {self.fingerprint.agent_id} cleaned up successfully")

        except Exception as e:
            logger.error(f"Error during agent cleanup: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.fingerprint.agent_id,
            "agent_type": self.fingerprint.agent_type,
            "status": self.fingerprint.status,
            "current_request": self.fingerprint.current_request_id,
            "performance": self.fingerprint.performance_metrics.__dict__,
            "capabilities": [cap.name for cap in self.fingerprint.capabilities]
        }

    def __str__(self) -> str:
        return f"{self.fingerprint.agent_type}Agent({self.fingerprint.agent_id})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(agent_id='{self.fingerprint.agent_id}', type='{self.fingerprint.agent_type}', status='{self.fingerprint.status}')"
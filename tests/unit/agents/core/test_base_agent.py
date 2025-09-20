"""
Unit Tests for BaseAgent Class
==============================

Tests for the base agent functionality including:
- Agent initialization and configuration
- Context management and state sharing
- Agent fingerprinting and identity
- Error handling and recovery
- Execution lifecycle
- Performance monitoring
"""

import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from agents.core.base_agent import (
    BaseAgent,
    AgentError,
    AgentTimeoutError,
    AgentValidationError
)
from agents.core.context_manager import SharedContext
from agents.core.fingerprinting import AgentType, AgentStatus


class TestAgentError:
    """Test agent error classes"""

    def test_agent_error_creation(self):
        """Test agent error creation"""
        error = AgentError("Test error", "agent-123", "TEST_ERROR")

        assert str(error) == "Test error"
        assert error.agent_id == "agent-123"
        assert error.error_code == "TEST_ERROR"
        assert isinstance(error.timestamp, datetime)

    def test_agent_error_default_code(self):
        """Test agent error with default error code"""
        error = AgentError("Test error", "agent-123")

        assert error.error_code == "AGENT_ERROR"

    def test_agent_timeout_error(self):
        """Test agent timeout error"""
        error = AgentTimeoutError("agent-123", 30.0)

        assert isinstance(error, AgentError)
        assert error.error_code == "TIMEOUT"
        assert error.timeout == 30.0

    def test_agent_validation_error(self):
        """Test agent validation error"""
        error = AgentValidationError("agent-123", "Validation failed")

        assert isinstance(error, AgentError)


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""

    async def _execute_impl(self, **kwargs) -> dict:
        """Implementation of abstract execute method"""
        await asyncio.sleep(0.01)  # Simulate work
        return {"result": "success", "input": kwargs}


class TestBaseAgent:
    """Test BaseAgent functionality"""

    @pytest.fixture
    def shared_context(self):
        """Create shared context for testing"""
        return SharedContext()

    @pytest.fixture
    def agent(self, shared_context):
        """Create concrete agent for testing"""
        return ConcreteAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING,
            version="1.0.0",
            timeout=30.0,
            max_retries=2
        )

    def test_agent_initialization(self, shared_context):
        """Test agent initialization"""
        agent = ConcreteAgent(
            context=shared_context,
            agent_type=AgentType.CODE_GENERATION,
            version="2.0.0",
            timeout=60.0,
            max_retries=5
        )

        assert agent.context == shared_context
        assert agent.fingerprint.agent_type == AgentType.CODE_GENERATION
        assert agent.fingerprint.version == "2.0.0"
        assert agent.timeout == 60.0
        assert agent.max_retries == 5
        assert agent.fingerprint is not None

    def test_agent_fingerprint_registration(self, agent):
        """Test that agent fingerprint is registered"""
        assert agent.fingerprint is not None
        assert agent.fingerprint.agent_id is not None
        assert agent.fingerprint.agent_type == AgentType.PLANNING

    def test_agent_context_access(self, agent, shared_context):
        """Test agent can access shared context"""
        # Add some data to context
        shared_context.update_artifact("test_data", {"value": 123}, agent.fingerprint.agent_id)

        # Agent should be able to access the data
        assert agent.context.get_artifact("test_data") == {"value": 123}

    @pytest.mark.asyncio
    async def test_agent_execute_success(self, agent):
        """Test successful agent execution"""
        result = await agent.execute(test_param="test_value")

        assert result["result"] == "success"
        assert result["input"]["test_param"] == "test_value"

    def test_agent_get_status(self, agent):
        """Test getting agent status"""
        status = agent.get_status()

        assert isinstance(status, dict)
        assert "agent_id" in status
        assert "agent_type" in status
        assert "status" in status
        assert "performance" in status

    def test_agent_string_representation(self, agent):
        """Test agent string representation"""
        str_repr = str(agent)
        assert agent.fingerprint.agent_id in str_repr
        assert agent.fingerprint.agent_type.value in str_repr

    def test_agent_repr(self, agent):
        """Test agent repr representation"""
        repr_str = repr(agent)
        assert "ConcreteAgent" in repr_str
        assert agent.fingerprint.agent_id in repr_str

    @pytest.mark.asyncio
    async def test_agent_cleanup(self, agent):
        """Test agent cleanup"""
        await agent.cleanup()
        # Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_agent_send_message(self, agent):
        """Test agent message sending"""
        message = {"type": "test", "data": "test_data"}
        await agent.send_message(message)
        # Should complete without error
        assert True

    def test_agent_validation_abstract_methods(self, shared_context):
        """Test that abstract methods must be implemented"""
        with pytest.raises(TypeError):
            # Cannot instantiate BaseAgent directly
            BaseAgent(shared_context, AgentType.PLANNING)


class SlowAgent(ConcreteAgent):
    """Agent that takes time to execute"""

    async def _execute_impl(self, **kwargs) -> dict:
        await asyncio.sleep(2.0)  # Longer than typical timeout
        return {"result": "slow_success"}


class FailingAgent(ConcreteAgent):
    """Agent that always fails"""

    async def _execute_impl(self, **kwargs) -> dict:
        raise ValueError("Test execution error")


class RetryAgent(ConcreteAgent):
    """Agent that fails then succeeds"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attempt_count = 0

    async def _execute_impl(self, **kwargs) -> dict:
        self.attempt_count += 1
        if self.attempt_count < 3:  # Fail first 2 attempts
            raise ValueError("Temporary failure")
        return {"result": "retry_success", "attempts": self.attempt_count}


class TestAgentExecutionScenarios:
    """Test various agent execution scenarios"""

    @pytest.fixture
    def shared_context(self):
        return SharedContext()

    @pytest.mark.asyncio
    async def test_agent_timeout_behavior(self, shared_context):
        """Test agent timeout behavior"""
        agent = SlowAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING,
            timeout=0.5  # Short timeout
        )

        with pytest.raises(AgentTimeoutError):
            await agent.execute(test_param="value")

        assert agent.fingerprint.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_agent_execution_error(self, shared_context):
        """Test agent execution error handling"""
        agent = FailingAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING
        )

        with pytest.raises(AgentError):
            await agent.execute(test_param="value")

        assert agent.fingerprint.status == AgentStatus.FAILED

    @pytest.mark.asyncio
    async def test_agent_retry_logic(self, shared_context):
        """Test agent retry logic"""
        agent = RetryAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING,
            max_retries=3
        )

        result = await agent.execute(test_param="value")

        assert result["result"] == "retry_success"
        assert result["attempts"] == 3
        assert agent.fingerprint.status == AgentStatus.COMPLETED


class TestAgentWithMocking:
    """Test agent functionality with mocking"""

    @pytest.fixture
    def shared_context(self):
        return SharedContext()

    @pytest.fixture
    def agent(self, shared_context):
        return ConcreteAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING
        )

    @pytest.mark.asyncio
    @patch('agents.core.base_agent.tracer')
    async def test_agent_tracing_integration(self, mock_tracer, agent):
        """Test agent tracing integration"""
        mock_span = Mock()
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span

        await agent.execute(test_param="value")

        mock_tracer.start_as_current_span.assert_called_once()
        mock_span.set_attribute.assert_called()

    @pytest.mark.asyncio
    @patch('agents.core.base_agent.psutil')
    async def test_agent_resource_monitoring(self, mock_psutil, agent):
        """Test agent resource monitoring"""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 1024 * 1024  # 1MB
        mock_process.cpu_percent.return_value = 25.0
        mock_psutil.Process.return_value = mock_process

        await agent.execute(test_param="value")

        # Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_agent_context_update(self, agent):
        """Test that agent updates context during execution"""
        initial_artifacts = len(agent.context.artifacts)

        await agent.execute(test_param="value")

        # Context should be updated (this depends on implementation)
        # At minimum, it should complete without error
        assert agent.fingerprint.status == AgentStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_agent_performance_tracking(self, agent):
        """Test agent performance tracking"""
        initial_executions = agent.fingerprint.performance_metrics.execution_count

        await agent.execute(test_param="value")

        # Performance metrics should be updated
        assert agent.fingerprint.performance_metrics.execution_count > initial_executions


class TestAgentErrorRecovery:
    """Test agent error recovery mechanisms"""

    @pytest.fixture
    def shared_context(self):
        return SharedContext()

    @pytest.mark.asyncio
    async def test_agent_error_metadata(self, shared_context):
        """Test error metadata collection"""
        agent = FailingAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING
        )

        try:
            await agent.execute(test_param="value")
        except AgentError as e:
            assert e.agent_id == agent.fingerprint.agent_id
            assert isinstance(e.timestamp, datetime)
        else:
            pytest.fail("Expected AgentError to be raised")

    @pytest.mark.asyncio
    async def test_agent_status_transitions(self, shared_context):
        """Test agent status transitions during execution"""
        agent = ConcreteAgent(
            context=shared_context,
            agent_type=AgentType.PLANNING
        )

        # Initial status
        assert agent.fingerprint.status == AgentStatus.INITIALIZING

        await agent.execute(test_param="value")

        # Final status
        assert agent.fingerprint.status == AgentStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__])
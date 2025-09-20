"""
Unit Tests for Context Manager
==============================

Tests for SharedContext and ContextManager functionality including:
- Context creation and persistence
- Decision and constraint management
- Checkpoint and rollback functionality
- Distributed locking
- Error handling
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from agents.core.context_manager import (
    SharedContext,
    ContextManager,
    Decision,
    Constraint,
    BusinessIntent,
    DecisionType,
    ConstraintType,
    get_context_manager
)
from agents.core.exceptions import ContextError


class TestSharedContext:
    """Test SharedContext functionality"""

    def test_context_creation(self):
        """Test basic context creation"""
        context = SharedContext()

        assert context.request_id is not None
        assert context.session_id is not None
        assert isinstance(context.business_intent, BusinessIntent)
        assert context.current_phase == "initialization"
        assert context.artifacts == {}
        assert context.decisions == []
        assert context.constraints == []

    def test_add_decision(self):
        """Test adding decisions to context"""
        context = SharedContext()

        decision = Decision(
            agent_id="test-agent",
            agent_type="planning",
            decision_type=DecisionType.BUSINESS_MODEL,
            description="Test decision",
            rationale="Test rationale",
            confidence=0.8
        )

        context.add_decision(decision)

        assert len(context.decisions) == 1
        assert context.decisions[0] == decision
        assert context.last_updated > context.start_time

    def test_add_constraint(self):
        """Test adding constraints to context"""
        context = SharedContext()

        constraint = Constraint(
            constraint_type=ConstraintType.BUSINESS,
            description="Budget constraint",
            priority=1,
            source="user"
        )

        context.add_constraint(constraint)

        assert len(context.constraints) == 1
        assert context.constraints[0] == constraint

    def test_create_checkpoint(self):
        """Test checkpoint creation"""
        context = SharedContext()
        context.current_phase = "planning"
        context.active_agents = ["agent1", "agent2"]

        checkpoint = context.create_checkpoint("test-agent", "Planning complete")

        assert len(context.checkpoints) == 1
        assert checkpoint.agent_id == "test-agent"
        assert checkpoint.description == "Planning complete"
        assert checkpoint.state_snapshot["current_phase"] == "planning"
        assert checkpoint.state_snapshot["active_agents"] == ["agent1", "agent2"]

    def test_get_decisions_by_type(self):
        """Test filtering decisions by type"""
        context = SharedContext()

        # Add different types of decisions
        decision1 = Decision(
            agent_id="agent1",
            agent_type="planning",
            decision_type=DecisionType.BUSINESS_MODEL,
            description="Business model decision"
        )

        decision2 = Decision(
            agent_id="agent2",
            agent_type="architecture",
            decision_type=DecisionType.ARCHITECTURE,
            description="Architecture decision"
        )

        decision3 = Decision(
            agent_id="agent3",
            agent_type="planning",
            decision_type=DecisionType.BUSINESS_MODEL,
            description="Another business model decision"
        )

        context.add_decision(decision1)
        context.add_decision(decision2)
        context.add_decision(decision3)

        business_decisions = context.get_decisions_by_type(DecisionType.BUSINESS_MODEL)
        architecture_decisions = context.get_decisions_by_type(DecisionType.ARCHITECTURE)

        assert len(business_decisions) == 2
        assert len(architecture_decisions) == 1
        assert business_decisions[0].description == "Business model decision"
        assert architecture_decisions[0].description == "Architecture decision"

    def test_update_artifact(self):
        """Test artifact updates"""
        context = SharedContext()

        context.update_artifact("test_key", {"data": "value"}, "test-agent")

        assert "test_key" in context.artifacts
        artifact = context.artifacts["test_key"]
        assert artifact["value"] == {"data": "value"}
        assert artifact["updated_by"] == "test-agent"
        assert "updated_at" in artifact

    def test_get_artifact(self):
        """Test artifact retrieval"""
        context = SharedContext()

        # Test non-existent artifact
        assert context.get_artifact("non-existent") is None

        # Test existing artifact
        context.update_artifact("test_key", "test_value", "test-agent")
        assert context.get_artifact("test_key") == "test_value"

    def test_context_serialization(self):
        """Test context serialization to dict"""
        context = SharedContext()
        context.original_request = "Test business"

        # Add some data
        decision = Decision(
            agent_id="test-agent",
            agent_type="planning",
            decision_type=DecisionType.BUSINESS_MODEL,
            description="Test decision"
        )
        context.add_decision(decision)

        context_dict = context.to_dict()

        assert isinstance(context_dict, dict)
        assert context_dict["request_id"] == context.request_id
        assert context_dict["original_request"] == "Test business"
        assert len(context_dict["decisions"]) == 1


class TestContextManager:
    """Test ContextManager functionality"""

    @pytest.fixture
    def context_manager(self):
        """Create context manager for testing"""
        manager = ContextManager()

        # Mock Redis for testing
        with patch('agents.core.context_manager.redis') as mock_redis:
            mock_redis_instance = AsyncMock()
            mock_redis.from_url.return_value = mock_redis_instance
            mock_redis_instance.ping.return_value = True

            manager._redis = mock_redis_instance
            return manager

    @pytest.mark.asyncio
    async def test_context_creation_and_retrieval(self, context_manager):
        """Test context creation and retrieval"""
        context = SharedContext()
        context.original_request = "Test business"

        # Mock Redis operations
        context_manager._redis.setex = AsyncMock()
        context_manager._redis.get = AsyncMock()

        # Test creation
        await context_manager.create_context(context)

        context_manager._redis.setex.assert_called_once()
        call_args = context_manager._redis.setex.call_args
        assert call_args[0][0] == f"codemind:context:{context.request_id}"

        # Test retrieval
        stored_data = json.dumps(context.to_dict(), default=str)
        context_manager._redis.get.return_value = stored_data

        retrieved_context = await context_manager.get_context(context.request_id)

        assert retrieved_context is not None
        assert retrieved_context.request_id == context.request_id
        assert retrieved_context.original_request == "Test business"

    @pytest.mark.asyncio
    async def test_context_update_with_locking(self, context_manager):
        """Test context update with distributed locking"""
        context = SharedContext()

        # Mock Redis operations for locking
        context_manager._redis.set = AsyncMock(return_value=True)  # Lock acquired
        context_manager._redis.setex = AsyncMock()
        context_manager._redis.delete = AsyncMock()

        await context_manager.update_context(context)

        # Verify lock was acquired and released
        context_manager._redis.set.assert_called_once()
        context_manager._redis.delete.assert_called_once()
        context_manager._redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_update_lock_failure(self, context_manager):
        """Test context update when lock acquisition fails"""
        context = SharedContext()

        # Mock lock failure
        context_manager._redis.set = AsyncMock(return_value=False)

        with pytest.raises(RuntimeError, match="Could not acquire lock"):
            await context_manager.update_context(context)

    @pytest.mark.asyncio
    async def test_context_not_found(self, context_manager):
        """Test retrieving non-existent context"""
        context_manager._redis.get = AsyncMock(return_value=None)

        result = await context_manager.get_context("non-existent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_context_deserialization_error(self, context_manager):
        """Test handling of context deserialization errors"""
        context_manager._redis.get = AsyncMock(return_value="invalid-json")

        result = await context_manager.get_context("test-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_context_deletion(self, context_manager):
        """Test context deletion"""
        context_manager._redis.delete = AsyncMock()

        await context_manager.delete_context("test-id")

        context_manager._redis.delete.assert_called_once_with("codemind:context:test-id")

    @pytest.mark.asyncio
    async def test_list_contexts(self, context_manager):
        """Test listing contexts"""
        mock_keys = ["codemind:context:id1", "codemind:context:id2"]
        context_manager._redis.keys = AsyncMock(return_value=mock_keys)

        result = await context_manager.list_contexts()

        assert result == ["id1", "id2"]

    @pytest.mark.asyncio
    async def test_extend_ttl(self, context_manager):
        """Test extending context TTL"""
        context_manager._redis.expire = AsyncMock()

        await context_manager.extend_ttl("test-id", 3600)

        context_manager._redis.expire.assert_called_once_with("codemind:context:test-id", 3600)


class TestBusinessIntent:
    """Test BusinessIntent functionality"""

    def test_business_intent_creation(self):
        """Test BusinessIntent creation"""
        intent = BusinessIntent(
            description="E-commerce platform",
            industry="E-commerce & Retail",
            business_model="B2C",
            target_users=["consumers", "small businesses"],
            key_features=["product catalog", "shopping cart", "payment processing"]
        )

        assert intent.description == "E-commerce platform"
        assert intent.industry == "E-commerce & Retail"
        assert intent.business_model == "B2C"
        assert len(intent.target_users) == 2
        assert len(intent.key_features) == 3


class TestDecision:
    """Test Decision functionality"""

    def test_decision_creation(self):
        """Test Decision creation"""
        decision = Decision(
            agent_id="planning-agent-123",
            agent_type="planning",
            decision_type=DecisionType.BUSINESS_MODEL,
            description="Selected SaaS business model",
            rationale="Best fit for recurring revenue and scalability",
            confidence=0.9,
            alternatives_considered=["B2C", "Marketplace", "B2B"]
        )

        assert decision.agent_id == "planning-agent-123"
        assert decision.decision_type == DecisionType.BUSINESS_MODEL
        assert decision.confidence == 0.9
        assert len(decision.alternatives_considered) == 3
        assert isinstance(decision.timestamp, datetime)


class TestConstraint:
    """Test Constraint functionality"""

    def test_constraint_creation(self):
        """Test Constraint creation"""
        constraint = Constraint(
            constraint_type=ConstraintType.REGULATORY,
            description="GDPR compliance required",
            priority=1,
            source="business_requirements",
            validation_rule="data_handling is GDPR_compliant"
        )

        assert constraint.constraint_type == ConstraintType.REGULATORY
        assert constraint.priority == 1
        assert constraint.is_satisfied is False
        assert constraint.validation_rule == "data_handling is GDPR_compliant"

    def test_constraint_validation(self):
        """Test constraint satisfaction tracking"""
        constraint = Constraint(
            constraint_type=ConstraintType.BUDGET,
            description="Budget under $10,000",
            priority=2
        )

        # Initially not satisfied
        assert constraint.is_satisfied is False

        # Mark as satisfied
        constraint.is_satisfied = True
        assert constraint.is_satisfied is True


@pytest.mark.asyncio
async def test_get_context_manager():
    """Test global context manager initialization"""
    with patch('agents.core.context_manager.context_manager') as mock_manager:
        mock_manager._redis = None
        mock_manager.initialize = AsyncMock()

        manager = await get_context_manager()

        # Should initialize if not already done
        mock_manager.initialize.assert_called_once()
        assert manager is mock_manager


if __name__ == "__main__":
    pytest.main([__file__])
"""
Unit Tests for Agent Fingerprinting System
==========================================

Tests for agent identification, tracking, and observability including:
- Agent fingerprint creation and validation
- Performance metrics tracking
- Agent capability management
- Fingerprint tracking functionality
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, AsyncMock

from agents.core.fingerprinting import (
    AgentFingerprint,
    AgentType,
    AgentStatus,
    PerformanceMetrics,
    AgentCapability,
    FingerprintGenerator,
    FingerprintTracker,
    fingerprint_tracker
)


class TestAgentFingerprint:
    """Test AgentFingerprint functionality"""

    def test_fingerprint_creation_with_defaults(self):
        """Test agent fingerprint creation with default values"""
        fingerprint = AgentFingerprint()

        assert fingerprint.agent_id is not None
        assert len(fingerprint.agent_id) > 0
        assert fingerprint.agent_type == AgentType.PLANNING
        assert fingerprint.status == AgentStatus.INITIALIZING
        assert isinstance(fingerprint.created_at, datetime)
        assert isinstance(fingerprint.performance_metrics, PerformanceMetrics)

    def test_fingerprint_creation_with_custom_values(self):
        """Test agent fingerprint creation with custom values"""
        custom_id = "test-agent-123"
        fingerprint = AgentFingerprint(
            agent_id=custom_id,
            agent_type=AgentType.CODE_GENERATION,
            status=AgentStatus.ACTIVE,
            parent_agent_id="parent-123"
        )

        assert fingerprint.agent_id == custom_id
        assert fingerprint.agent_type == AgentType.CODE_GENERATION
        assert fingerprint.status == AgentStatus.ACTIVE
        assert fingerprint.parent_agent_id == "parent-123"

    def test_start_execution(self):
        """Test starting agent execution"""
        fingerprint = AgentFingerprint()
        request_id = "req-123"

        fingerprint.start_execution(request_id)

        assert fingerprint.status == AgentStatus.ACTIVE
        assert fingerprint.current_request_id == request_id

    def test_end_execution_success(self):
        """Test ending execution successfully"""
        fingerprint = AgentFingerprint()
        fingerprint.start_execution("req-123")

        fingerprint.end_execution(
            success=True,
            execution_time=5.5,
            tokens=150,
            cost=0.25
        )

        assert fingerprint.status == AgentStatus.COMPLETED
        assert fingerprint.current_request_id is None
        assert fingerprint.performance_metrics.last_execution_time == 5.5
        assert fingerprint.performance_metrics.tokens_consumed == 150
        assert fingerprint.performance_metrics.cost_incurred == 0.25

    def test_end_execution_failure(self):
        """Test ending execution with failure"""
        fingerprint = AgentFingerprint()
        fingerprint.start_execution("req-123")

        initial_errors = fingerprint.performance_metrics.error_count

        fingerprint.end_execution(
            success=False,
            execution_time=2.0,
            tokens=50,
            cost=0.1
        )

        assert fingerprint.status == AgentStatus.FAILED
        assert fingerprint.performance_metrics.error_count == initial_errors + 1

    def test_add_child_agent(self):
        """Test adding child agents to fingerprint"""
        parent = AgentFingerprint()
        child1_id = "child-1"
        child2_id = "child-2"

        parent.add_child_agent(child1_id)
        parent.add_child_agent(child2_id)

        assert len(parent.child_agent_ids) == 2
        assert child1_id in parent.child_agent_ids
        assert child2_id in parent.child_agent_ids

    def test_add_duplicate_child_agent(self):
        """Test adding duplicate child agent doesn't create duplicates"""
        parent = AgentFingerprint()
        child_id = "child-1"

        parent.add_child_agent(child_id)
        parent.add_child_agent(child_id)  # Add same child again

        assert len(parent.child_agent_ids) == 1
        assert child_id in parent.child_agent_ids

    def test_get_summary(self):
        """Test getting agent fingerprint summary"""
        fingerprint = AgentFingerprint(agent_id="test-123")
        fingerprint.start_execution("req-123")
        fingerprint.end_execution(True, 5.0, 0.5, 100)

        summary = fingerprint.get_summary()

        assert summary["agent_id"] == "test-123"
        assert summary["agent_type"] == AgentType.PLANNING
        assert summary["status"] == AgentStatus.COMPLETED
        assert summary["performance"]["total_cost"] == 0.5

    def test_fingerprint_serialization(self):
        """Test fingerprint serialization to dict"""
        fingerprint = AgentFingerprint(
            agent_id="test-123",
            agent_type=AgentType.ARCHITECTURE
        )
        fingerprint.add_child_agent("child-1")

        result_dict = fingerprint.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["agent_id"] == "test-123"
        assert result_dict["agent_type"] == AgentType.ARCHITECTURE
        assert "child-1" in result_dict["child_agent_ids"]


class TestPerformanceMetrics:
    """Test PerformanceMetrics functionality"""

    def test_performance_metrics_defaults(self):
        """Test performance metrics default values"""
        metrics = PerformanceMetrics()

        assert metrics.execution_count == 0
        assert metrics.total_execution_time == 0.0
        assert metrics.average_execution_time == 0.0
        assert metrics.success_rate == 0.0
        assert metrics.error_count == 0
        assert metrics.tokens_consumed == 0
        assert metrics.cost_incurred == 0.0

    def test_performance_metrics_with_values(self):
        """Test performance metrics with custom values"""
        metrics = PerformanceMetrics(
            execution_count=5,
            total_execution_time=25.0,
            tokens_consumed=1000,
            cost_incurred=5.50
        )

        assert metrics.execution_count == 5
        assert metrics.total_execution_time == 25.0
        assert metrics.tokens_consumed == 1000
        assert metrics.cost_incurred == 5.50


class TestAgentCapability:
    """Test AgentCapability functionality"""

    def test_capability_creation(self):
        """Test agent capability creation"""
        capability = AgentCapability(
            name="code_generation",
            description="Generate code from specifications",
            version="1.0.0",
            supported_inputs=["specification", "requirements"],
            supported_outputs=["code", "documentation"],
            dependencies=["llm_service", "template_engine"]
        )

        assert capability.name == "code_generation"
        assert capability.description == "Generate code from specifications"
        assert capability.version == "1.0.0"
        assert len(capability.supported_inputs) == 2
        assert len(capability.supported_outputs) == 2
        assert len(capability.dependencies) == 2

    def test_capability_defaults(self):
        """Test capability creation with defaults"""
        capability = AgentCapability(
            name="test_capability",
            description="Test description",
            version="1.0.0"
        )

        assert capability.supported_inputs == []
        assert capability.supported_outputs == []
        assert capability.dependencies == []
        assert capability.performance_profile == {}


class TestFingerprintGenerator:
    """Test FingerprintGenerator functionality"""

    def test_create_planning_agent_fingerprint(self):
        """Test creating planning agent fingerprint"""
        fingerprint = FingerprintGenerator.create_planning_agent_fingerprint()

        assert fingerprint.agent_type == AgentType.PLANNING
        assert fingerprint.agent_id is not None
        assert len(fingerprint.capabilities) > 0

    def test_create_architecture_agent_fingerprint(self):
        """Test creating architecture agent fingerprint"""
        fingerprint = FingerprintGenerator.create_architecture_agent_fingerprint(
            parent_agent_id="parent-456"
        )

        assert fingerprint.agent_type == AgentType.ARCHITECTURE
        assert fingerprint.parent_agent_id == "parent-456"

    def test_create_code_agent_fingerprint(self):
        """Test creating code generation agent fingerprint"""
        fingerprint = FingerprintGenerator.create_code_agent_fingerprint()

        assert fingerprint.agent_type == AgentType.CODE_GENERATION
        assert fingerprint.agent_id is not None

    def test_create_test_agent_fingerprint(self):
        """Test creating testing agent fingerprint"""
        fingerprint = FingerprintGenerator.create_test_agent_fingerprint()

        assert fingerprint.agent_type == AgentType.TESTING
        assert fingerprint.agent_id is not None

    def test_create_fingerprint_with_custom_params(self):
        """Test creating fingerprint with custom parameters"""
        fingerprint = FingerprintGenerator.create_fingerprint(
            agent_type=AgentType.TESTING,
            version="2.0.0",
            parent_agent_id="parent-123"
        )

        assert fingerprint.agent_type == AgentType.TESTING
        assert fingerprint.version == "2.0.0"
        assert fingerprint.parent_agent_id == "parent-123"


class TestFingerprintTracker:
    """Test FingerprintTracker functionality"""

    def test_tracker_initialization(self):
        """Test tracker initialization"""
        tracker = FingerprintTracker()

        assert len(tracker._active_fingerprints) == 0
        assert len(tracker._fingerprint_history) == 0

    def test_register_fingerprint(self):
        """Test registering a fingerprint"""
        tracker = FingerprintTracker()
        fingerprint = AgentFingerprint(agent_id="test-123")

        tracker.register_fingerprint(fingerprint)

        assert len(tracker._active_fingerprints) == 1
        assert "test-123" in tracker._active_fingerprints
        assert tracker._active_fingerprints["test-123"] == fingerprint

    def test_get_fingerprint_existing(self):
        """Test getting existing fingerprint"""
        tracker = FingerprintTracker()
        fingerprint = AgentFingerprint(agent_id="test-123")
        tracker.register_fingerprint(fingerprint)

        result = tracker.get_fingerprint("test-123")

        assert result is not None
        assert result.agent_id == "test-123"

    def test_get_fingerprint_non_existing(self):
        """Test getting non-existing fingerprint"""
        tracker = FingerprintTracker()

        result = tracker.get_fingerprint("non-existent")

        assert result is None

    def test_unregister_fingerprint(self):
        """Test unregistering a fingerprint"""
        tracker = FingerprintTracker()
        fingerprint = AgentFingerprint(agent_id="test-123")
        tracker.register_fingerprint(fingerprint)

        tracker.unregister_fingerprint("test-123")

        assert len(tracker._active_fingerprints) == 0
        assert len(tracker._fingerprint_history) == 1
        assert tracker._fingerprint_history[0] == fingerprint

    def test_get_active_fingerprints(self):
        """Test getting all active fingerprints"""
        tracker = FingerprintTracker()
        fp1 = AgentFingerprint(agent_id="test-1")
        fp2 = AgentFingerprint(agent_id="test-2")

        tracker.register_fingerprint(fp1)
        tracker.register_fingerprint(fp2)

        active = tracker.get_active_fingerprints()

        assert len(active) == 2
        assert fp1 in active
        assert fp2 in active

    def test_get_fingerprints_by_type(self):
        """Test getting fingerprints by agent type"""
        tracker = FingerprintTracker()

        planning_agent = AgentFingerprint(
            agent_id="planning-1",
            agent_type=AgentType.PLANNING
        )
        code_agent = AgentFingerprint(
            agent_id="code-1",
            agent_type=AgentType.CODE_GENERATION
        )
        planning_agent2 = AgentFingerprint(
            agent_id="planning-2",
            agent_type=AgentType.PLANNING
        )

        tracker.register_fingerprint(planning_agent)
        tracker.register_fingerprint(code_agent)
        tracker.register_fingerprint(planning_agent2)

        planning_results = tracker.get_fingerprints_by_type(AgentType.PLANNING)
        code_results = tracker.get_fingerprints_by_type(AgentType.CODE_GENERATION)

        assert len(planning_results) == 2
        assert len(code_results) == 1
        assert planning_agent in planning_results
        assert planning_agent2 in planning_results
        assert code_agent in code_results

    def test_get_performance_summary_empty(self):
        """Test getting performance summary with no agents"""
        tracker = FingerprintTracker()

        summary = tracker.get_performance_summary()

        assert summary["active_agents"] == 0
        assert summary["total_agents_created"] == 0
        assert summary["total_executions"] == 0
        assert summary["total_cost"] == 0
        assert summary["average_success_rate"] == 0

    def test_get_performance_summary_with_agents(self):
        """Test getting performance summary with agents"""
        tracker = FingerprintTracker()

        # Create and register fingerprints with some performance data
        fp1 = AgentFingerprint(agent_id="test-1")
        fp1.start_execution("req-1")
        fp1.end_execution(True, 5.0, 1.0, 100)

        fp2 = AgentFingerprint(agent_id="test-2")
        fp2.start_execution("req-2")
        fp2.end_execution(True, 3.0, 0.5, 50)

        tracker.register_fingerprint(fp1)
        tracker.register_fingerprint(fp2)

        summary = tracker.get_performance_summary()

        assert summary["active_agents"] == 2
        assert summary["total_agents_created"] == 2
        assert summary["total_cost"] == 1.5


class TestGlobalTracker:
    """Test global tracker functionality"""

    def test_global_fingerprint_tracker_exists(self):
        """Test that global tracker exists"""
        assert fingerprint_tracker is not None
        assert isinstance(fingerprint_tracker, FingerprintTracker)

    def test_global_tracker_registration(self):
        """Test registering with global tracker"""
        # Clear any existing data first
        fingerprint_tracker._active_fingerprints.clear()
        fingerprint_tracker._fingerprint_history.clear()

        fingerprint = AgentFingerprint(agent_id="global-test")
        fingerprint_tracker.register_fingerprint(fingerprint)

        assert len(fingerprint_tracker._active_fingerprints) >= 1
        assert fingerprint_tracker.get_fingerprint("global-test") is not None

        # Clean up
        fingerprint_tracker.unregister_fingerprint("global-test")


if __name__ == "__main__":
    pytest.main([__file__])
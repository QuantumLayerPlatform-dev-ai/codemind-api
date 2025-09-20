"""
Agent Fingerprinting System for CodeMind
========================================

Provides unique identification, tracking, and observability for all agents
in the cognitive software factory. Every agent action is traceable and attributable.
"""

import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio

try:
    from ...core.config import get_settings
    from ...core.logging import get_logger
except ImportError:
    from core.config import get_settings
    from core.logging import get_logger

logger = get_logger("fingerprinting")
settings = get_settings()


class AgentType(str, Enum):
    """Types of agents in the system"""
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"
    DOCUMENTATION = "documentation"
    EVOLUTION = "evolution"
    MONITORING = "monitoring"
    HEALING = "healing"


class AgentStatus(str, Enum):
    """Agent execution status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    SUSPENDED = "suspended"
    TERMINATED = "terminated"


@dataclass
class AgentCapability:
    """Represents a capability of an agent"""
    name: str
    description: str
    version: str
    supported_inputs: List[str] = field(default_factory=list)
    supported_outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    performance_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an agent"""
    execution_count: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    last_execution_time: float = 0.0
    last_execution_timestamp: Optional[datetime] = None
    memory_usage: Dict[str, float] = field(default_factory=dict)
    cpu_usage: Dict[str, float] = field(default_factory=dict)
    tokens_consumed: int = 0
    cost_incurred: float = 0.0


@dataclass
class AgentFingerprint:
    """
    Unique fingerprint for an agent instance.
    Provides complete traceability and attribution.
    """
    # Core Identity
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_type: AgentType = AgentType.PLANNING
    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Version Information
    version: str = "1.0.0"
    build_hash: str = ""
    framework_version: str = "0.1.0"

    # Capabilities
    capabilities: List[AgentCapability] = field(default_factory=list)
    supported_models: List[str] = field(default_factory=list)
    max_concurrency: int = 1

    # Configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)

    # Runtime Information
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: AgentStatus = AgentStatus.INITIALIZING
    current_request_id: Optional[str] = None
    parent_agent_id: Optional[str] = None
    child_agent_ids: List[str] = field(default_factory=list)

    # Performance
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)

    # System Information
    host_info: Dict[str, str] = field(default_factory=dict)
    python_version: str = field(default_factory=lambda: platform.python_version())

    # Tracing
    trace_id: Optional[str] = None
    span_id: Optional[str] = None

    # Metadata
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize derived fields"""
        if not self.build_hash:
            self.build_hash = self._generate_build_hash()

        if not self.host_info:
            self.host_info = self._get_host_info()

    def _generate_build_hash(self) -> str:
        """Generate a unique hash for this agent build"""
        hash_data = {
            "agent_type": self.agent_type,
            "version": self.version,
            "framework_version": self.framework_version,
            "capabilities": [cap.name for cap in self.capabilities],
            "model_config": self.model_config,
            "python_version": self.python_version
        }
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_string.encode()).hexdigest()[:16]

    def _get_host_info(self) -> Dict[str, str]:
        """Get system information"""
        return {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture()[0]
        }

    def add_capability(self, capability: AgentCapability) -> None:
        """Add a capability to this agent"""
        self.capabilities.append(capability)
        logger.info(f"Capability added to {self.agent_id}: {capability.name}")

    def has_capability(self, capability_name: str) -> bool:
        """Check if agent has a specific capability"""
        return any(cap.name == capability_name for cap in self.capabilities)

    def update_status(self, status: AgentStatus, metadata: Dict[str, Any] = None) -> None:
        """Update agent status"""
        old_status = self.status
        self.status = status

        if metadata:
            self.metadata.update(metadata)

        logger.info(f"Agent {self.agent_id} status: {old_status} -> {status}")

    def start_execution(self, request_id: str, trace_id: str = None, span_id: str = None) -> None:
        """Mark start of execution"""
        self.current_request_id = request_id
        self.trace_id = trace_id
        self.span_id = span_id
        self.status = AgentStatus.ACTIVE

        # Update performance metrics
        self.performance_metrics.execution_count += 1

        logger.info(f"Agent {self.agent_id} started execution for request {request_id}")

    def end_execution(self, success: bool = True, execution_time: float = 0.0,
                     cost: float = 0.0, tokens: int = 0) -> None:
        """Mark end of execution"""
        # Update performance metrics
        metrics = self.performance_metrics
        metrics.last_execution_time = execution_time
        metrics.last_execution_timestamp = datetime.now(timezone.utc)
        metrics.total_execution_time += execution_time
        metrics.average_execution_time = metrics.total_execution_time / metrics.execution_count
        metrics.cost_incurred += cost
        metrics.tokens_consumed += tokens

        if success:
            metrics.success_rate = (metrics.success_rate * (metrics.execution_count - 1) + 1.0) / metrics.execution_count
            self.status = AgentStatus.COMPLETED
        else:
            metrics.error_count += 1
            metrics.success_rate = (metrics.success_rate * (metrics.execution_count - 1)) / metrics.execution_count
            self.status = AgentStatus.FAILED

        self.current_request_id = None

        logger.info(f"Agent {self.agent_id} execution completed: success={success}, time={execution_time:.2f}s")

    def add_child_agent(self, child_agent_id: str) -> None:
        """Add a child agent ID"""
        if child_agent_id not in self.child_agent_ids:
            self.child_agent_ids.append(child_agent_id)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of this agent's fingerprint"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "version": self.version,
            "status": self.status,
            "capabilities": [cap.name for cap in self.capabilities],
            "performance": {
                "executions": self.performance_metrics.execution_count,
                "success_rate": self.performance_metrics.success_rate,
                "avg_time": self.performance_metrics.average_execution_time,
                "total_cost": self.performance_metrics.cost_incurred
            },
            "created_at": self.created_at.isoformat(),
            "build_hash": self.build_hash
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "instance_id": self.instance_id,
            "version": self.version,
            "build_hash": self.build_hash,
            "framework_version": self.framework_version,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "version": cap.version,
                    "supported_inputs": cap.supported_inputs,
                    "supported_outputs": cap.supported_outputs,
                    "dependencies": cap.dependencies,
                    "performance_profile": cap.performance_profile
                }
                for cap in self.capabilities
            ],
            "supported_models": self.supported_models,
            "max_concurrency": self.max_concurrency,
            "model_config": self.model_config,
            "parameters": self.parameters,
            "environment": self.environment,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "current_request_id": self.current_request_id,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": self.child_agent_ids,
            "performance_metrics": {
                "execution_count": self.performance_metrics.execution_count,
                "total_execution_time": self.performance_metrics.total_execution_time,
                "average_execution_time": self.performance_metrics.average_execution_time,
                "success_rate": self.performance_metrics.success_rate,
                "error_count": self.performance_metrics.error_count,
                "last_execution_time": self.performance_metrics.last_execution_time,
                "last_execution_timestamp": self.performance_metrics.last_execution_timestamp.isoformat() if self.performance_metrics.last_execution_timestamp else None,
                "memory_usage": self.performance_metrics.memory_usage,
                "cpu_usage": self.performance_metrics.cpu_usage,
                "tokens_consumed": self.performance_metrics.tokens_consumed,
                "cost_incurred": self.performance_metrics.cost_incurred
            },
            "host_info": self.host_info,
            "python_version": self.python_version,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "tags": list(self.tags),
            "metadata": self.metadata
        }


class FingerprintGenerator:
    """
    Generates and manages agent fingerprints.
    Provides factory methods for creating fingerprints for different agent types.
    """

    @staticmethod
    def create_fingerprint(
        agent_type: AgentType,
        version: str = "1.0.0",
        capabilities: List[AgentCapability] = None,
        model_config: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None,
        parent_agent_id: str = None,
        tags: Set[str] = None
    ) -> AgentFingerprint:
        """Create a new agent fingerprint"""

        fingerprint = AgentFingerprint(
            agent_type=agent_type,
            version=version,
            capabilities=capabilities or [],
            model_config=model_config or {},
            parameters=parameters or {},
            parent_agent_id=parent_agent_id,
            tags=tags or set()
        )

        # Add default capabilities based on agent type
        default_capabilities = FingerprintGenerator._get_default_capabilities(agent_type)
        fingerprint.capabilities.extend(default_capabilities)

        logger.info(f"Created fingerprint for {agent_type} agent: {fingerprint.agent_id}")
        return fingerprint

    @staticmethod
    def _get_default_capabilities(agent_type: AgentType) -> List[AgentCapability]:
        """Get default capabilities for an agent type"""
        capabilities_map = {
            AgentType.PLANNING: [
                AgentCapability(
                    name="business_analysis",
                    description="Analyze business requirements and extract technical specifications",
                    version="1.0",
                    supported_inputs=["business_description", "user_requirements"],
                    supported_outputs=["technical_specifications", "feature_list"]
                ),
                AgentCapability(
                    name="constraint_identification",
                    description="Identify business and technical constraints",
                    version="1.0",
                    supported_inputs=["business_context"],
                    supported_outputs=["constraints_list"]
                )
            ],
            AgentType.ARCHITECTURE: [
                AgentCapability(
                    name="system_design",
                    description="Design system architecture based on requirements",
                    version="1.0",
                    supported_inputs=["technical_specifications", "constraints"],
                    supported_outputs=["architecture_diagram", "component_specifications"]
                ),
                AgentCapability(
                    name="technology_selection",
                    description="Select appropriate technologies and frameworks",
                    version="1.0",
                    supported_inputs=["requirements", "constraints"],
                    supported_outputs=["technology_stack", "justification"]
                )
            ],
            AgentType.CODE_GENERATION: [
                AgentCapability(
                    name="code_synthesis",
                    description="Generate code from specifications",
                    version="1.0",
                    supported_inputs=["specifications", "architecture"],
                    supported_outputs=["source_code", "configuration_files"]
                ),
                AgentCapability(
                    name="template_processing",
                    description="Process and customize code templates",
                    version="1.0",
                    supported_inputs=["templates", "parameters"],
                    supported_outputs=["customized_code"]
                )
            ],
            AgentType.TESTING: [
                AgentCapability(
                    name="test_generation",
                    description="Generate comprehensive test suites",
                    version="1.0",
                    supported_inputs=["source_code", "specifications"],
                    supported_outputs=["test_code", "test_coverage_report"]
                )
            ]
        }

        return capabilities_map.get(agent_type, [])

    @staticmethod
    def create_planning_agent_fingerprint(**kwargs) -> AgentFingerprint:
        """Create fingerprint for planning agent"""
        return FingerprintGenerator.create_fingerprint(AgentType.PLANNING, **kwargs)

    @staticmethod
    def create_architecture_agent_fingerprint(**kwargs) -> AgentFingerprint:
        """Create fingerprint for architecture agent"""
        return FingerprintGenerator.create_fingerprint(AgentType.ARCHITECTURE, **kwargs)

    @staticmethod
    def create_code_agent_fingerprint(**kwargs) -> AgentFingerprint:
        """Create fingerprint for code generation agent"""
        return FingerprintGenerator.create_fingerprint(AgentType.CODE_GENERATION, **kwargs)

    @staticmethod
    def create_test_agent_fingerprint(**kwargs) -> AgentFingerprint:
        """Create fingerprint for testing agent"""
        return FingerprintGenerator.create_fingerprint(AgentType.TESTING, **kwargs)


class FingerprintTracker:
    """
    Tracks and monitors agent fingerprints throughout their lifecycle.
    Provides observability and analytics capabilities.
    """

    def __init__(self):
        self._active_fingerprints: Dict[str, AgentFingerprint] = {}
        self._fingerprint_history: List[AgentFingerprint] = []

    def register_fingerprint(self, fingerprint: AgentFingerprint) -> None:
        """Register a new agent fingerprint"""
        self._active_fingerprints[fingerprint.agent_id] = fingerprint
        logger.info(f"Registered fingerprint: {fingerprint.agent_id}")

    def unregister_fingerprint(self, agent_id: str) -> None:
        """Unregister an agent fingerprint"""
        if agent_id in self._active_fingerprints:
            fingerprint = self._active_fingerprints.pop(agent_id)
            self._fingerprint_history.append(fingerprint)
            logger.info(f"Unregistered fingerprint: {agent_id}")

    def get_fingerprint(self, agent_id: str) -> Optional[AgentFingerprint]:
        """Get an active fingerprint by agent ID"""
        return self._active_fingerprints.get(agent_id)

    def get_active_fingerprints(self) -> List[AgentFingerprint]:
        """Get all active fingerprints"""
        return list(self._active_fingerprints.values())

    def get_fingerprints_by_type(self, agent_type: AgentType) -> List[AgentFingerprint]:
        """Get all active fingerprints of a specific type"""
        return [fp for fp in self._active_fingerprints.values() if fp.agent_type == agent_type]

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all agents"""
        active_agents = list(self._active_fingerprints.values())
        historical_agents = self._fingerprint_history

        total_executions = sum(
            fp.performance_metrics.execution_count
            for fp in active_agents + historical_agents
        )

        total_cost = sum(
            fp.performance_metrics.cost_incurred
            for fp in active_agents + historical_agents
        )

        average_success_rate = sum(
            fp.performance_metrics.success_rate
            for fp in active_agents + historical_agents
        ) / len(active_agents + historical_agents) if (active_agents + historical_agents) else 0

        return {
            "active_agents": len(active_agents),
            "total_agents_created": len(active_agents) + len(historical_agents),
            "total_executions": total_executions,
            "total_cost": total_cost,
            "average_success_rate": average_success_rate,
            "agent_types": {
                agent_type.value: len([fp for fp in active_agents if fp.agent_type == agent_type])
                for agent_type in AgentType
            }
        }


# Global fingerprint tracker
fingerprint_tracker = FingerprintTracker()
"""
Agent Registry for CodeMind Cognitive Software Factory
=====================================================

Central registry for agent discovery, version management, and capability tracking.
Provides service discovery and dynamic agent orchestration capabilities.
"""

import asyncio
from typing import Dict, List, Optional, Set, Type, Any, Callable
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from ...core.logging import get_logger
    from ...core.config import get_settings
except ImportError:
    from core.logging import get_logger
    from core.config import get_settings

from .fingerprinting import AgentFingerprint, AgentType, AgentStatus
from .base_agent import BaseAgent

logger = get_logger("agent_registry")
settings = get_settings()


class RegistrationStatus(str, Enum):
    """Agent registration status"""
    REGISTERED = "registered"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEREGISTERED = "deregistered"


@dataclass
class AgentRegistration:
    """Agent registration record"""
    fingerprint: AgentFingerprint
    agent_class: Type[BaseAgent]
    factory_function: Optional[Callable] = None
    registration_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: RegistrationStatus = RegistrationStatus.REGISTERED
    health_score: float = 1.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    """
    Central registry for all agents in the cognitive software factory.

    Provides:
    - Agent discovery and lookup
    - Version management
    - Capability-based routing
    - Health monitoring
    - Dynamic scaling decisions
    """

    def __init__(self):
        self._registrations: Dict[str, AgentRegistration] = {}
        self._type_index: Dict[AgentType, List[str]] = {}
        self._capability_index: Dict[str, List[str]] = {}
        self._version_index: Dict[str, Dict[str, List[str]]] = {}
        self._health_scores: Dict[str, float] = {}

        # Initialize indexes
        for agent_type in AgentType:
            self._type_index[agent_type] = []

    def register_agent_class(
        self,
        agent_class: Type[BaseAgent],
        agent_type: AgentType,
        version: str = "1.0.0",
        factory_function: Callable = None,
        tags: Set[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Register an agent class for dynamic instantiation.

        Args:
            agent_class: The agent class to register
            agent_type: Type of agent
            version: Agent version
            factory_function: Optional factory function for custom instantiation
            tags: Optional tags for categorization
            metadata: Optional metadata

        Returns:
            Registration ID
        """
        from .fingerprinting import FingerprintGenerator

        # Create a representative fingerprint for the class
        fingerprint = FingerprintGenerator.create_fingerprint(
            agent_type=agent_type,
            version=version,
            tags=tags or set()
        )

        registration = AgentRegistration(
            fingerprint=fingerprint,
            agent_class=agent_class,
            factory_function=factory_function,
            tags=tags or set(),
            metadata=metadata or {}
        )

        registration_id = fingerprint.agent_id
        self._registrations[registration_id] = registration

        # Update indexes
        self._update_indexes(registration_id, registration)

        logger.info(f"Registered agent class: {agent_class.__name__} as {agent_type} v{version}")
        return registration_id

    def register_agent_instance(self, agent: BaseAgent) -> str:
        """
        Register an active agent instance.

        Args:
            agent: The agent instance to register

        Returns:
            Registration ID
        """
        registration = AgentRegistration(
            fingerprint=agent.fingerprint,
            agent_class=type(agent),
            status=RegistrationStatus.ACTIVE
        )

        registration_id = agent.fingerprint.agent_id
        self._registrations[registration_id] = registration

        # Update indexes
        self._update_indexes(registration_id, registration)

        logger.info(f"Registered agent instance: {registration_id}")
        return registration_id

    def deregister_agent(self, agent_id: str) -> bool:
        """
        Deregister an agent.

        Args:
            agent_id: ID of the agent to deregister

        Returns:
            True if successful, False if agent not found
        """
        if agent_id not in self._registrations:
            logger.warning(f"Attempted to deregister unknown agent: {agent_id}")
            return False

        registration = self._registrations[agent_id]
        registration.status = RegistrationStatus.DEREGISTERED

        # Remove from indexes
        self._remove_from_indexes(agent_id, registration)

        # Remove from registry
        del self._registrations[agent_id]

        logger.info(f"Deregistered agent: {agent_id}")
        return True

    def get_agent_registration(self, agent_id: str) -> Optional[AgentRegistration]:
        """Get agent registration by ID"""
        return self._registrations.get(agent_id)

    def get_agents_by_type(self, agent_type: AgentType) -> List[AgentRegistration]:
        """Get all registered agents of a specific type"""
        agent_ids = self._type_index.get(agent_type, [])
        return [self._registrations[aid] for aid in agent_ids if aid in self._registrations]

    def get_agents_by_capability(self, capability: str) -> List[AgentRegistration]:
        """Get all agents that have a specific capability"""
        agent_ids = self._capability_index.get(capability, [])
        return [self._registrations[aid] for aid in agent_ids if aid in self._registrations]

    def get_agents_by_version(self, agent_type: AgentType, version: str) -> List[AgentRegistration]:
        """Get all agents of a specific type and version"""
        if agent_type.value not in self._version_index:
            return []

        agent_ids = self._version_index[agent_type.value].get(version, [])
        return [self._registrations[aid] for aid in agent_ids if aid in self._registrations]

    def get_best_agent_for_task(
        self,
        agent_type: AgentType,
        required_capabilities: List[str] = None,
        preferred_version: str = None,
        exclude_busy: bool = True
    ) -> Optional[AgentRegistration]:
        """
        Find the best agent for a specific task.

        Args:
            agent_type: Required agent type
            required_capabilities: List of required capabilities
            preferred_version: Preferred version (optional)
            exclude_busy: Whether to exclude busy agents

        Returns:
            Best matching agent registration or None
        """
        candidates = self.get_agents_by_type(agent_type)

        if not candidates:
            logger.warning(f"No agents found for type: {agent_type}")
            return None

        # Filter by capabilities
        if required_capabilities:
            candidates = [
                reg for reg in candidates
                if all(
                    any(cap.name == req_cap for cap in reg.fingerprint.capabilities)
                    for req_cap in required_capabilities
                )
            ]

        if not candidates:
            logger.warning(f"No agents found with required capabilities: {required_capabilities}")
            return None

        # Filter by status
        candidates = [
            reg for reg in candidates
            if reg.status in [RegistrationStatus.REGISTERED, RegistrationStatus.ACTIVE]
        ]

        # Exclude busy agents if requested
        if exclude_busy:
            candidates = [
                reg for reg in candidates
                if reg.fingerprint.status not in [AgentStatus.ACTIVE, AgentStatus.WAITING]
            ]

        if not candidates:
            logger.warning("No available agents found")
            return None

        # Prefer specific version if requested
        if preferred_version:
            version_matches = [
                reg for reg in candidates
                if reg.fingerprint.version == preferred_version
            ]
            if version_matches:
                candidates = version_matches

        # Sort by health score and performance
        candidates.sort(
            key=lambda reg: (
                reg.health_score,
                reg.fingerprint.performance_metrics.success_rate,
                -reg.fingerprint.performance_metrics.average_execution_time
            ),
            reverse=True
        )

        best_agent = candidates[0]
        logger.info(f"Selected best agent: {best_agent.fingerprint.agent_id} for {agent_type}")
        return best_agent

    def create_agent_instance(
        self,
        agent_type: AgentType,
        context,
        required_capabilities: List[str] = None,
        preferred_version: str = None,
        **kwargs
    ) -> Optional[BaseAgent]:
        """
        Create an agent instance for immediate use.

        Args:
            agent_type: Type of agent to create
            context: Shared context for the agent
            required_capabilities: Required capabilities
            preferred_version: Preferred version
            **kwargs: Additional arguments for agent creation

        Returns:
            Agent instance or None if creation fails
        """
        registration = self.get_best_agent_for_task(
            agent_type=agent_type,
            required_capabilities=required_capabilities,
            preferred_version=preferred_version,
            exclude_busy=False  # Allow creating new instances
        )

        if not registration:
            logger.error(f"No suitable agent registration found for type: {agent_type}")
            return None

        try:
            # Create agent instance
            if registration.factory_function:
                agent = registration.factory_function(context=context, **kwargs)
            else:
                agent = registration.agent_class(
                    context=context,
                    agent_type=agent_type,
                    **kwargs
                )

            # Register the instance
            self.register_agent_instance(agent)

            logger.info(f"Created agent instance: {agent.fingerprint.agent_id}")
            return agent

        except Exception as e:
            logger.error(f"Failed to create agent instance: {e}")
            return None

    def update_agent_health(self, agent_id: str, health_score: float) -> None:
        """Update health score for an agent"""
        if agent_id in self._registrations:
            self._registrations[agent_id].health_score = max(0.0, min(1.0, health_score))
            self._registrations[agent_id].last_activity = datetime.now(timezone.utc)
            logger.debug(f"Updated health score for {agent_id}: {health_score}")

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        total_agents = len(self._registrations)
        active_agents = len([
            reg for reg in self._registrations.values()
            if reg.status == RegistrationStatus.ACTIVE
        ])

        type_counts = {}
        for agent_type in AgentType:
            type_counts[agent_type.value] = len(self._type_index.get(agent_type, []))

        capability_counts = {
            cap: len(agents) for cap, agents in self._capability_index.items()
        }

        average_health = sum(
            reg.health_score for reg in self._registrations.values()
        ) / total_agents if total_agents > 0 else 0

        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "type_distribution": type_counts,
            "capability_distribution": capability_counts,
            "average_health_score": average_health,
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def list_all_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with their details"""
        return [
            {
                "agent_id": reg.fingerprint.agent_id,
                "agent_type": reg.fingerprint.agent_type,
                "version": reg.fingerprint.version,
                "status": reg.status,
                "health_score": reg.health_score,
                "capabilities": [cap.name for cap in reg.fingerprint.capabilities],
                "registration_time": reg.registration_time.isoformat(),
                "last_activity": reg.last_activity.isoformat(),
                "performance": {
                    "executions": reg.fingerprint.performance_metrics.execution_count,
                    "success_rate": reg.fingerprint.performance_metrics.success_rate,
                    "avg_time": reg.fingerprint.performance_metrics.average_execution_time,
                    "total_cost": reg.fingerprint.performance_metrics.cost_incurred
                }
            }
            for reg in self._registrations.values()
        ]

    def _update_indexes(self, agent_id: str, registration: AgentRegistration) -> None:
        """Update all indexes with new registration"""
        fingerprint = registration.fingerprint

        # Type index
        if agent_id not in self._type_index[fingerprint.agent_type]:
            self._type_index[fingerprint.agent_type].append(agent_id)

        # Capability index
        for capability in fingerprint.capabilities:
            if capability.name not in self._capability_index:
                self._capability_index[capability.name] = []
            if agent_id not in self._capability_index[capability.name]:
                self._capability_index[capability.name].append(agent_id)

        # Version index
        agent_type_key = fingerprint.agent_type.value
        if agent_type_key not in self._version_index:
            self._version_index[agent_type_key] = {}
        if fingerprint.version not in self._version_index[agent_type_key]:
            self._version_index[agent_type_key][fingerprint.version] = []
        if agent_id not in self._version_index[agent_type_key][fingerprint.version]:
            self._version_index[agent_type_key][fingerprint.version].append(agent_id)

    def _remove_from_indexes(self, agent_id: str, registration: AgentRegistration) -> None:
        """Remove agent from all indexes"""
        fingerprint = registration.fingerprint

        # Type index
        if agent_id in self._type_index[fingerprint.agent_type]:
            self._type_index[fingerprint.agent_type].remove(agent_id)

        # Capability index
        for capability in fingerprint.capabilities:
            if capability.name in self._capability_index:
                if agent_id in self._capability_index[capability.name]:
                    self._capability_index[capability.name].remove(agent_id)

        # Version index
        agent_type_key = fingerprint.agent_type.value
        if agent_type_key in self._version_index:
            if fingerprint.version in self._version_index[agent_type_key]:
                if agent_id in self._version_index[agent_type_key][fingerprint.version]:
                    self._version_index[agent_type_key][fingerprint.version].remove(agent_id)


# Global agent registry instance
agent_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry"""
    return agent_registry


# Decorator for auto-registration
def register_agent(
    agent_type: AgentType,
    version: str = "1.0.0",
    tags: Set[str] = None,
    metadata: Dict[str, Any] = None
):
    """
    Decorator to automatically register agent classes.

    Usage:
        @register_agent(AgentType.PLANNING, version="1.1.0")
        class PlanningAgent(BaseAgent):
            pass
    """
    def decorator(agent_class: Type[BaseAgent]):
        # Register the class
        agent_registry.register_agent_class(
            agent_class=agent_class,
            agent_type=agent_type,
            version=version,
            tags=tags,
            metadata=metadata
        )
        return agent_class

    return decorator
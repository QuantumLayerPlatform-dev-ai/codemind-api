"""
SharedContext Management for CodeMind Agents
===========================================

Provides distributed context sharing across all agents in the cognitive software factory.
Context includes business intent, decisions, artifacts, and state needed for collaboration.
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio

import redis.asyncio as redis
from pydantic import BaseModel, Field

try:
    from ...core.config import get_settings
    from ...core.logging import get_logger
except ImportError:
    from core.config import get_settings
    from core.logging import get_logger

logger = get_logger("context_manager")
settings = get_settings()


class DecisionType(str, Enum):
    """Types of decisions made by agents"""
    BUSINESS_MODEL = "business_model"
    ARCHITECTURE = "architecture"
    TECHNOLOGY_CHOICE = "technology_choice"
    FEATURE_SELECTION = "feature_selection"
    DEPLOYMENT_STRATEGY = "deployment_strategy"
    SECURITY_POLICY = "security_policy"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"


class ConstraintType(str, Enum):
    """Types of constraints"""
    BUSINESS = "business"
    TECHNICAL = "technical"
    REGULATORY = "regulatory"
    BUDGET = "budget"
    TIMELINE = "timeline"
    SECURITY = "security"
    PERFORMANCE = "performance"


@dataclass
class Decision:
    """Represents a decision made by an agent"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    agent_type: str = ""
    decision_type: DecisionType = DecisionType.BUSINESS_MODEL
    description: str = ""
    rationale: str = ""
    confidence: float = 0.0
    alternatives_considered: List[str] = field(default_factory=list)
    impact_assessment: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Constraint:
    """Represents a constraint that must be satisfied"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    constraint_type: ConstraintType = ConstraintType.BUSINESS
    description: str = ""
    priority: int = 1  # 1=critical, 2=high, 3=medium, 4=low
    source: str = ""  # Where the constraint came from
    validation_rule: str = ""  # How to validate this constraint
    is_satisfied: bool = False
    violation_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Represents a checkpoint for rollback capability"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    description: str = ""
    state_snapshot: Dict[str, Any] = field(default_factory=dict)
    artifacts_snapshot: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_rollback_point: bool = True


@dataclass
class BusinessIntent:
    """Extracted business intent"""
    description: str = ""
    industry: str = ""
    business_model: str = ""
    target_users: List[str] = field(default_factory=list)
    key_features: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    compliance_requirements: List[str] = field(default_factory=list)
    budget_constraints: Dict[str, Any] = field(default_factory=dict)
    timeline_constraints: Dict[str, Any] = field(default_factory=dict)
    technical_preferences: Dict[str, Any] = field(default_factory=dict)
    competitive_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedContext:
    """
    Shared context that persists across all agent interactions.
    This is the central nervous system of the cognitive software factory.
    """
    # Identity
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None

    # Business Understanding
    business_intent: BusinessIntent = field(default_factory=BusinessIntent)
    original_request: str = ""
    complexity_score: float = 0.5

    # Collaboration State
    artifacts: Dict[str, Any] = field(default_factory=dict)  # Shared artifacts
    decisions: List[Decision] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    checkpoints: List[Checkpoint] = field(default_factory=list)

    # Execution State
    current_phase: str = "initialization"
    active_agents: List[str] = field(default_factory=list)
    completed_agents: List[str] = field(default_factory=list)
    failed_agents: List[str] = field(default_factory=list)

    # Metrics
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    total_cost: float = 0.0
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_decision(self, decision: Decision) -> None:
        """Add a decision to the context"""
        self.decisions.append(decision)
        self.last_updated = datetime.now(timezone.utc)
        logger.info(f"Decision added: {decision.description} by {decision.agent_type}")

    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the context"""
        self.constraints.append(constraint)
        self.last_updated = datetime.now(timezone.utc)
        logger.info(f"Constraint added: {constraint.description}")

    def create_checkpoint(self, agent_id: str, description: str) -> Checkpoint:
        """Create a checkpoint for rollback"""
        checkpoint = Checkpoint(
            agent_id=agent_id,
            description=description,
            state_snapshot={
                "current_phase": self.current_phase,
                "active_agents": self.active_agents.copy(),
                "completed_agents": self.completed_agents.copy()
            },
            artifacts_snapshot=self.artifacts.copy()
        )
        self.checkpoints.append(checkpoint)
        self.last_updated = datetime.now(timezone.utc)
        logger.info(f"Checkpoint created: {description}")
        return checkpoint

    def get_decisions_by_type(self, decision_type: DecisionType) -> List[Decision]:
        """Get all decisions of a specific type"""
        return [d for d in self.decisions if d.decision_type == decision_type]

    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[Constraint]:
        """Get all constraints of a specific type"""
        return [c for c in self.constraints if c.constraint_type == constraint_type]

    def get_unsatisfied_constraints(self) -> List[Constraint]:
        """Get all constraints that are not yet satisfied"""
        return [c for c in self.constraints if not c.is_satisfied]

    def update_artifact(self, key: str, value: Any, agent_id: str) -> None:
        """Update an artifact in the shared context"""
        self.artifacts[key] = {
            "value": value,
            "updated_by": agent_id,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        self.last_updated = datetime.now(timezone.utc)
        logger.info(f"Artifact updated: {key} by {agent_id}")

    def get_artifact(self, key: str) -> Optional[Any]:
        """Get an artifact from the shared context"""
        artifact = self.artifacts.get(key)
        return artifact["value"] if artifact else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class ContextManager:
    """
    Manages shared context storage and retrieval using Redis.
    Provides distributed context sharing across all agents.
    """

    def __init__(self):
        self._redis: Optional[redis.Redis] = None
        self._context_ttl = 86400  # 24 hours
        self._lock_timeout = 30  # 30 seconds for locks

    async def initialize(self) -> None:
        """Initialize Redis connection"""
        try:
            redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379')
            self._redis = redis.from_url(redis_url, decode_responses=True)
            await self._redis.ping()
            logger.info("ContextManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContextManager: {e}")
            raise

    async def close(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()

    def _context_key(self, request_id: str) -> str:
        """Generate Redis key for context"""
        return f"codemind:context:{request_id}"

    def _lock_key(self, request_id: str) -> str:
        """Generate Redis key for context lock"""
        return f"codemind:context:lock:{request_id}"

    async def create_context(self, context: SharedContext) -> None:
        """Create new shared context"""
        if not self._redis:
            raise RuntimeError("ContextManager not initialized")

        key = self._context_key(context.request_id)
        context_data = json.dumps(context.to_dict(), default=str)

        await self._redis.setex(key, self._context_ttl, context_data)
        logger.info(f"Context created: {context.request_id}")

    async def get_context(self, request_id: str) -> Optional[SharedContext]:
        """Retrieve shared context"""
        if not self._redis:
            raise RuntimeError("ContextManager not initialized")

        key = self._context_key(request_id)
        context_data = await self._redis.get(key)

        if not context_data:
            logger.warning(f"Context not found: {request_id}")
            return None

        try:
            data = json.loads(context_data)

            # Convert back to SharedContext
            # Handle datetime fields
            if 'start_time' in data and isinstance(data['start_time'], str):
                data['start_time'] = datetime.fromisoformat(data['start_time'])
            if 'last_updated' in data and isinstance(data['last_updated'], str):
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])

            # Convert decisions
            decisions = []
            for d in data.get('decisions', []):
                if isinstance(d['timestamp'], str):
                    d['timestamp'] = datetime.fromisoformat(d['timestamp'])
                decisions.append(Decision(**d))
            data['decisions'] = decisions

            # Convert constraints
            constraints = []
            for c in data.get('constraints', []):
                constraints.append(Constraint(**c))
            data['constraints'] = constraints

            # Convert checkpoints
            checkpoints = []
            for cp in data.get('checkpoints', []):
                if isinstance(cp['timestamp'], str):
                    cp['timestamp'] = datetime.fromisoformat(cp['timestamp'])
                checkpoints.append(Checkpoint(**cp))
            data['checkpoints'] = checkpoints

            # Convert business intent
            if 'business_intent' in data:
                data['business_intent'] = BusinessIntent(**data['business_intent'])

            return SharedContext(**data)

        except Exception as e:
            logger.error(f"Failed to deserialize context {request_id}: {e}")
            return None

    async def update_context(self, context: SharedContext) -> None:
        """Update shared context with distributed locking"""
        if not self._redis:
            raise RuntimeError("ContextManager not initialized")

        lock_key = self._lock_key(context.request_id)
        context_key = self._context_key(context.request_id)

        # Acquire distributed lock
        lock_acquired = await self._redis.set(
            lock_key, "locked", nx=True, ex=self._lock_timeout
        )

        if not lock_acquired:
            logger.warning(f"Failed to acquire lock for context: {context.request_id}")
            # Wait briefly and retry once
            await asyncio.sleep(0.1)
            lock_acquired = await self._redis.set(
                lock_key, "locked", nx=True, ex=self._lock_timeout
            )

            if not lock_acquired:
                raise RuntimeError(f"Could not acquire lock for context: {context.request_id}")

        try:
            # Update last_updated timestamp
            context.last_updated = datetime.now(timezone.utc)

            # Serialize and store
            context_data = json.dumps(context.to_dict(), default=str)
            await self._redis.setex(context_key, self._context_ttl, context_data)

            logger.debug(f"Context updated: {context.request_id}")

        finally:
            # Release lock
            await self._redis.delete(lock_key)

    async def delete_context(self, request_id: str) -> None:
        """Delete shared context"""
        if not self._redis:
            raise RuntimeError("ContextManager not initialized")

        key = self._context_key(request_id)
        await self._redis.delete(key)
        logger.info(f"Context deleted: {request_id}")

    async def list_contexts(self, pattern: str = "codemind:context:*") -> List[str]:
        """List all context keys matching pattern"""
        if not self._redis:
            raise RuntimeError("ContextManager not initialized")

        keys = await self._redis.keys(pattern)
        return [key.split(":")[-1] for key in keys]  # Extract request IDs

    async def extend_ttl(self, request_id: str, ttl: int = None) -> None:
        """Extend TTL for a context"""
        if not self._redis:
            raise RuntimeError("ContextManager not initialized")

        key = self._context_key(request_id)
        ttl = ttl or self._context_ttl
        await self._redis.expire(key, ttl)


# Global context manager instance
context_manager = ContextManager()


async def get_context_manager() -> ContextManager:
    """Get initialized context manager"""
    if context_manager._redis is None:
        await context_manager.initialize()
    return context_manager
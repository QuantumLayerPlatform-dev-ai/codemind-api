"""
CodeMind Agent Framework
========================

The cognitive software factory multi-agent system that understands business intent
and generates complete, deployable, self-evolving applications.

This package provides:
- Core agent framework with fingerprinting and context sharing
- Cognitive agents for planning, architecture, code generation, and testing
- Specialized agents for security, performance, and compliance
- Orchestration and workflow management
"""

from .core.base_agent import BaseAgent
from .core.context_manager import SharedContext, ContextManager
from .core.fingerprinting import AgentFingerprint, FingerprintGenerator
from .core.registry import AgentRegistry

__version__ = "0.1.0"
__all__ = [
    "BaseAgent",
    "SharedContext",
    "ContextManager",
    "AgentFingerprint",
    "FingerprintGenerator",
    "AgentRegistry"
]
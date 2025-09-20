"""
Temporal Activities for CodeMind Agent Execution
===============================================

Activities wrap individual agent executions to provide:
- Retry logic
- Timeout handling
- State persistence
- Error recovery
"""

import asyncio
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from temporalio import activity

try:
    from ..core.logging import get_logger
    from ..agents.core.context_manager import SharedContext, get_context_manager
    from ..agents.core.registry import get_agent_registry
    from ..agents.core.fingerprinting import AgentType
except ImportError:
    from core.logging import get_logger
    from agents.core.context_manager import SharedContext, get_context_manager
    from agents.core.registry import get_agent_registry
    from agents.core.fingerprinting import AgentType

logger = get_logger("temporal_activities")


async def execute_agent_activity(
    context_dict: Dict[str, Any],
    agent_type: AgentType,
    agent_kwargs: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Generic activity for executing any agent type.

    Args:
        context_dict: Serialized shared context
        agent_type: Type of agent to execute
        agent_kwargs: Additional arguments for agent creation

    Returns:
        Activity result with updated context
    """
    activity.logger.info(f"Executing {agent_type} agent activity")

    try:
        # Reconstruct context
        context = SharedContext(**context_dict)

        # Get agent registry
        registry = get_agent_registry()

        # Create agent instance
        agent = registry.create_agent_instance(
            agent_type=agent_type,
            context=context,
            **(agent_kwargs or {})
        )

        if not agent:
            raise RuntimeError(f"Failed to create {agent_type} agent")

        # Initialize agent
        await agent.initialize()

        # Execute agent
        result = await agent.execute()

        # Get updated context
        updated_context = agent.context

        # Cleanup agent
        await agent.cleanup()

        activity.logger.info(f"{agent_type} agent executed successfully")

        return {
            "success": True,
            "agent_type": agent_type.value,
            "agent_id": agent.fingerprint.agent_id,
            "result": result,
            "context": updated_context.to_dict(),
            "execution_time": agent.fingerprint.performance_metrics.last_execution_time,
            "cost": agent.fingerprint.performance_metrics.cost_incurred
        }

    except Exception as e:
        activity.logger.error(f"{agent_type} agent activity failed: {e}")
        return {
            "success": False,
            "agent_type": agent_type.value,
            "error": str(e),
            "context": context_dict  # Return original context on failure
        }


@activity.defn
async def planning_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute planning agent to analyze business intent and create technical specifications.

    Args:
        context_dict: Serialized shared context

    Returns:
        Planning results with updated context
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.PLANNING
    )


@activity.defn
async def architecture_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute architecture agent to design system architecture.

    Args:
        context_dict: Serialized shared context

    Returns:
        Architecture results with updated context
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.ARCHITECTURE
    )


@activity.defn
async def code_generation_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute code generation agent to create application code.

    Args:
        context_dict: Serialized shared context

    Returns:
        Code generation results with updated context
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.CODE_GENERATION
    )


@activity.defn
async def testing_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute testing agent to generate and run tests.

    Args:
        context_dict: Serialized shared context

    Returns:
        Testing results with updated context
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.TESTING
    )


@activity.defn
async def validation_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute validation agent to verify all constraints are satisfied.

    Args:
        context_dict: Serialized shared context

    Returns:
        Validation results with updated context
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.VALIDATION
    )


@activity.defn
async def deployment_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute deployment agent to deploy the generated application.

    Args:
        context_dict: Serialized shared context

    Returns:
        Deployment results with updated context
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.DEPLOYMENT
    )


@activity.defn
async def monitoring_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute monitoring agent to set up application monitoring.

    Args:
        context_dict: Serialized shared context

    Returns:
        Monitoring setup results with updated context
    """
    if isinstance(context_dict, dict) and "application_id" in context_dict:
        # This is a health check call
        return await _check_application_health(context_dict)

    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.MONITORING
    )


async def _check_application_health(params: Dict[str, Any]) -> Dict[str, Any]:
    """Check health of a deployed application"""
    application_id = params["application_id"]

    try:
        # TODO: Implement actual health checking logic
        # This would typically involve:
        # - HTTP health check endpoints
        # - Database connectivity checks
        # - Resource usage monitoring
        # - Error rate analysis

        activity.logger.info(f"Checking health for application: {application_id}")

        # Placeholder implementation
        health_status = {
            "healthy": True,
            "status_code": 200,
            "response_time_ms": 150,
            "issues": []
        }

        return health_status

    except Exception as e:
        activity.logger.error(f"Health check failed for {application_id}: {e}")
        return {
            "healthy": False,
            "error": str(e),
            "issues": ["health_check_failed"]
        }


@activity.defn
async def security_scan_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute security scanning on generated code.

    Args:
        context_dict: Serialized shared context

    Returns:
        Security scan results
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.SECURITY
    )


@activity.defn
async def performance_optimization_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute performance optimization agent.

    Args:
        context_dict: Serialized shared context

    Returns:
        Performance optimization results
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.PERFORMANCE
    )


@activity.defn
async def compliance_check_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute compliance checking agent.

    Args:
        context_dict: Serialized shared context

    Returns:
        Compliance check results
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.COMPLIANCE
    )


@activity.defn
async def documentation_generation_activity(context_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute documentation generation agent.

    Args:
        context_dict: Serialized shared context

    Returns:
        Documentation generation results
    """
    return await execute_agent_activity(
        context_dict=context_dict,
        agent_type=AgentType.DOCUMENTATION
    )


@activity.defn
async def parallel_agent_execution_activity(
    context_dict: Dict[str, Any],
    agent_types: list,
    timeout_seconds: int = 300
) -> Dict[str, Any]:
    """
    Execute multiple agents in parallel.

    Args:
        context_dict: Serialized shared context
        agent_types: List of agent types to execute in parallel
        timeout_seconds: Timeout for parallel execution

    Returns:
        Combined results from all agents
    """
    activity.logger.info(f"Executing agents in parallel: {agent_types}")

    try:
        # Create tasks for each agent
        tasks = []
        for agent_type_str in agent_types:
            agent_type = AgentType(agent_type_str)
            task = execute_agent_activity(context_dict, agent_type)
            tasks.append((agent_type, task))

        # Execute all tasks with timeout
        results = {}
        completed_tasks = await asyncio.wait_for(
            asyncio.gather(
                *[task for _, task in tasks],
                return_exceptions=True
            ),
            timeout=timeout_seconds
        )

        # Process results
        for i, (agent_type, _) in enumerate(tasks):
            result = completed_tasks[i]
            if isinstance(result, Exception):
                results[agent_type.value] = {
                    "success": False,
                    "error": str(result)
                }
            else:
                results[agent_type.value] = result

        # Merge contexts (use the most recently updated one)
        final_context = context_dict
        latest_update = datetime.min.replace(tzinfo=timezone.utc)

        for agent_result in results.values():
            if agent_result.get("success") and "context" in agent_result:
                context_data = agent_result["context"]
                if "last_updated" in context_data:
                    update_time = datetime.fromisoformat(context_data["last_updated"])
                    if update_time > latest_update:
                        latest_update = update_time
                        final_context = context_data

        return {
            "success": True,
            "parallel_results": results,
            "context": final_context,
            "agents_executed": len([r for r in results.values() if r.get("success")]),
            "agents_failed": len([r for r in results.values() if not r.get("success")]),
            "total_execution_time": max(
                r.get("execution_time", 0) for r in results.values() if r.get("success")
            )
        }

    except asyncio.TimeoutError:
        activity.logger.error(f"Parallel agent execution timed out after {timeout_seconds} seconds")
        return {
            "success": False,
            "error": "timeout",
            "context": context_dict
        }

    except Exception as e:
        activity.logger.error(f"Parallel agent execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "context": context_dict
        }
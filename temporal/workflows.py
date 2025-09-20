"""
Temporal Workflows for CodeMind Agent Orchestration
===================================================

Defines workflows that orchestrate multiple agents to complete
complex software generation tasks.
"""

import asyncio
from datetime import timedelta
from typing import Any, Dict, List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

try:
    from ..core.logging import get_logger
    from ..agents.core.context_manager import SharedContext, BusinessIntent
    from ..agents.core.fingerprinting import AgentType
except ImportError:
    from core.logging import get_logger
    from agents.core.context_manager import SharedContext, BusinessIntent
    from agents.core.fingerprinting import AgentType

from .activities import (
    planning_activity,
    architecture_activity,
    code_generation_activity,
    testing_activity,
    validation_activity,
    deployment_activity,
    monitoring_activity
)

logger = get_logger("temporal_workflows")


@workflow.defn
class CodeGenerationWorkflow:
    """
    Main workflow for generating complete applications from business intent.

    This workflow orchestrates all agents in the cognitive software factory
    to transform a business description into a deployed, working application.
    """

    @workflow.run
    async def run(
        self,
        business_description: str,
        complexity: float = 0.5,
        user_requirements: Dict[str, Any] = None,
        constraints: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete code generation workflow.

        Args:
            business_description: Natural language description of the business
            complexity: Complexity score (0.0-1.0)
            user_requirements: Additional user requirements
            constraints: Business and technical constraints

        Returns:
            Complete generation results including deployed application details
        """
        workflow.logger.info(f"Starting code generation workflow for: {business_description}")

        # Create shared context for all agents
        context = SharedContext(
            request_id=workflow.info().workflow_id,
            original_request=business_description,
            complexity_score=complexity,
            business_intent=BusinessIntent(description=business_description)
        )

        # If user provided additional constraints, add them to context
        if constraints:
            context.metadata["user_constraints"] = constraints

        if user_requirements:
            context.metadata["user_requirements"] = user_requirements

        # Define retry policy for activities
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=60),
            maximum_attempts=3
        )

        try:
            # Phase 1: Planning - Understand business intent and create technical plan
            workflow.logger.info("Phase 1: Planning")
            planning_result = await workflow.execute_activity(
                planning_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry_policy
            )

            # Update context with planning results
            context = SharedContext(**planning_result["context"])
            context.current_phase = "architecture"

            # Phase 2: Architecture - Design system architecture
            workflow.logger.info("Phase 2: Architecture Design")
            architecture_result = await workflow.execute_activity(
                architecture_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy
            )

            # Update context with architecture results
            context = SharedContext(**architecture_result["context"])
            context.current_phase = "code_generation"

            # Phase 3: Code Generation - Generate actual code
            workflow.logger.info("Phase 3: Code Generation")
            code_result = await workflow.execute_activity(
                code_generation_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=15),
                retry_policy=retry_policy
            )

            # Update context with code generation results
            context = SharedContext(**code_result["context"])
            context.current_phase = "testing"

            # Phase 4: Testing - Generate and run tests
            workflow.logger.info("Phase 4: Testing")
            test_result = await workflow.execute_activity(
                testing_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=10),
                retry_policy=retry_policy
            )

            # Update context with test results
            context = SharedContext(**test_result["context"])
            context.current_phase = "validation"

            # Phase 5: Validation - Validate all constraints are met
            workflow.logger.info("Phase 5: Validation")
            validation_result = await workflow.execute_activity(
                validation_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry_policy
            )

            # Update context with validation results
            context = SharedContext(**validation_result["context"])
            context.current_phase = "deployment"

            # Phase 6: Deployment - Deploy the application
            workflow.logger.info("Phase 6: Deployment")
            deployment_result = await workflow.execute_activity(
                deployment_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=20),
                retry_policy=retry_policy
            )

            # Update context with deployment results
            context = SharedContext(**deployment_result["context"])
            context.current_phase = "monitoring"

            # Phase 7: Setup Monitoring - Configure monitoring and health checks
            workflow.logger.info("Phase 7: Monitoring Setup")
            monitoring_result = await workflow.execute_activity(
                monitoring_activity,
                context.to_dict(),
                start_to_close_timeout=timedelta(minutes=5),
                retry_policy=retry_policy
            )

            # Final context update
            context = SharedContext(**monitoring_result["context"])
            context.current_phase = "completed"

            # Compile final results
            final_result = {
                "success": True,
                "request_id": context.request_id,
                "business_description": business_description,
                "generated_artifacts": context.artifacts,
                "deployment_info": context.get_artifact("deployment_info"),
                "application_url": context.get_artifact("application_url"),
                "monitoring_dashboard": context.get_artifact("monitoring_dashboard"),
                "decisions_made": [decision.__dict__ for decision in context.decisions],
                "constraints_satisfied": all(c.is_satisfied for c in context.constraints),
                "total_cost": context.total_cost,
                "execution_time": (context.last_updated - context.start_time).total_seconds(),
                "agents_involved": context.completed_agents,
                "performance_metrics": context.performance_metrics,
                "context": context.to_dict()
            }

            workflow.logger.info(f"Code generation workflow completed successfully: {context.request_id}")
            return final_result

        except Exception as e:
            workflow.logger.error(f"Code generation workflow failed: {e}")

            # Create failure result
            failure_result = {
                "success": False,
                "request_id": context.request_id,
                "error": str(e),
                "phase_failed": context.current_phase,
                "partial_artifacts": context.artifacts,
                "decisions_made": [decision.__dict__ for decision in context.decisions],
                "total_cost": context.total_cost,
                "agents_involved": context.completed_agents + context.failed_agents,
                "context": context.to_dict()
            }

            return failure_result


@workflow.defn
class AgentOrchestrationWorkflow:
    """
    Flexible workflow for orchestrating arbitrary agent combinations.

    This workflow allows for dynamic agent composition and execution
    based on runtime requirements.
    """

    @workflow.run
    async def run(
        self,
        context_dict: Dict[str, Any],
        agent_sequence: List[AgentType],
        parallel_execution: bool = False,
        timeout_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Execute a custom sequence of agents.

        Args:
            context_dict: Serialized shared context
            agent_sequence: List of agent types to execute
            parallel_execution: Whether to execute agents in parallel
            timeout_minutes: Overall timeout for the workflow

        Returns:
            Execution results
        """
        workflow.logger.info(f"Starting agent orchestration workflow with agents: {agent_sequence}")

        context = SharedContext(**context_dict)

        # Define activity mapping
        activity_map = {
            AgentType.PLANNING: planning_activity,
            AgentType.ARCHITECTURE: architecture_activity,
            AgentType.CODE_GENERATION: code_generation_activity,
            AgentType.TESTING: testing_activity,
            AgentType.VALIDATION: validation_activity,
            AgentType.DEPLOYMENT: deployment_activity,
            AgentType.MONITORING: monitoring_activity
        }

        # Retry policy
        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(seconds=30),
            maximum_attempts=2
        )

        results = {}

        try:
            if parallel_execution:
                # Execute agents in parallel
                workflow.logger.info("Executing agents in parallel")

                tasks = []
                for agent_type in agent_sequence:
                    if agent_type in activity_map:
                        task = workflow.execute_activity(
                            activity_map[agent_type],
                            context.to_dict(),
                            start_to_close_timeout=timedelta(minutes=timeout_minutes // len(agent_sequence)),
                            retry_policy=retry_policy
                        )
                        tasks.append((agent_type, task))

                # Wait for all tasks to complete
                for agent_type, task in tasks:
                    try:
                        result = await task
                        results[agent_type.value] = result
                        # Update context with latest state
                        if "context" in result:
                            context = SharedContext(**result["context"])
                    except Exception as e:
                        workflow.logger.error(f"Agent {agent_type} failed: {e}")
                        results[agent_type.value] = {"error": str(e)}

            else:
                # Execute agents sequentially
                workflow.logger.info("Executing agents sequentially")

                for agent_type in agent_sequence:
                    if agent_type in activity_map:
                        try:
                            workflow.logger.info(f"Executing {agent_type}")
                            result = await workflow.execute_activity(
                                activity_map[agent_type],
                                context.to_dict(),
                                start_to_close_timeout=timedelta(minutes=timeout_minutes // len(agent_sequence)),
                                retry_policy=retry_policy
                            )

                            results[agent_type.value] = result

                            # Update context for next agent
                            if "context" in result:
                                context = SharedContext(**result["context"])

                        except Exception as e:
                            workflow.logger.error(f"Agent {agent_type} failed: {e}")
                            results[agent_type.value] = {"error": str(e)}
                            # Continue with other agents even if one fails

            # Compile final results
            final_result = {
                "success": True,
                "request_id": context.request_id,
                "agent_results": results,
                "final_context": context.to_dict(),
                "execution_summary": {
                    "agents_executed": len([r for r in results.values() if "error" not in r]),
                    "agents_failed": len([r for r in results.values() if "error" in r]),
                    "total_cost": context.total_cost,
                    "execution_time": (context.last_updated - context.start_time).total_seconds()
                }
            }

            workflow.logger.info("Agent orchestration workflow completed")
            return final_result

        except Exception as e:
            workflow.logger.error(f"Agent orchestration workflow failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "partial_results": results,
                "context": context.to_dict()
            }


@workflow.defn
class SelfHealingWorkflow:
    """
    Workflow for self-healing and continuous improvement of deployed applications.
    """

    @workflow.run
    async def run(
        self,
        application_id: str,
        monitoring_interval_minutes: int = 5,
        healing_enabled: bool = True
    ) -> Dict[str, Any]:
        """
        Continuously monitor and heal an application.

        Args:
            application_id: ID of the deployed application
            monitoring_interval_minutes: How often to check health
            healing_enabled: Whether to attempt automatic healing

        Returns:
            Monitoring and healing results
        """
        workflow.logger.info(f"Starting self-healing workflow for application: {application_id}")

        healing_attempts = 0
        max_healing_attempts = 5

        while healing_attempts < max_healing_attempts:
            try:
                # Wait for monitoring interval
                await asyncio.sleep(monitoring_interval_minutes * 60)

                # Check application health
                health_result = await workflow.execute_activity(
                    monitoring_activity,
                    {"application_id": application_id, "check_health": True},
                    start_to_close_timeout=timedelta(minutes=2)
                )

                if health_result.get("healthy", True):
                    workflow.logger.info(f"Application {application_id} is healthy")
                    continue

                # Application is unhealthy
                workflow.logger.warning(f"Application {application_id} is unhealthy: {health_result}")

                if healing_enabled:
                    # Attempt healing
                    healing_result = await workflow.execute_activity(
                        "healing_activity",  # TODO: Implement healing activity
                        {
                            "application_id": application_id,
                            "health_issues": health_result.get("issues", [])
                        },
                        start_to_close_timeout=timedelta(minutes=10)
                    )

                    healing_attempts += 1

                    if healing_result.get("success", False):
                        workflow.logger.info(f"Successfully healed application {application_id}")
                        healing_attempts = 0  # Reset counter on successful healing
                    else:
                        workflow.logger.error(f"Failed to heal application {application_id}")

                else:
                    # Just alert, don't heal
                    workflow.logger.warning(f"Healing disabled for application {application_id}")
                    break

            except Exception as e:
                workflow.logger.error(f"Self-healing workflow error: {e}")
                healing_attempts += 1

        return {
            "application_id": application_id,
            "healing_attempts": healing_attempts,
            "status": "completed" if healing_attempts < max_healing_attempts else "max_attempts_reached"
        }
"""
Enterprise Health Checks and Circuit Breakers for CodeMind Agents
================================================================

Comprehensive health monitoring system with:
- Service health checks
- Circuit breaker patterns
- Performance monitoring
- Automatic failover
- Health dashboards
"""

import asyncio
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from dataclasses import dataclass, field
import statistics

import aiohttp
import psutil
from circuitbreaker import circuit

try:
    from ...core.logging import get_logger
    from .config import get_settings
    from .exceptions import (
        AgentResourceError,
        ExternalServiceError,
        NetworkError,
        ErrorSeverity
    )
except ImportError:
    from core.logging import get_logger
    from agents.core.config import get_settings
    from agents.core.exceptions import (
        AgentResourceError,
        ExternalServiceError,
        NetworkError,
        ErrorSeverity
    )

logger = get_logger("health_checks")


class HealthStatus(str, Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ServiceType(str, Enum):
    """Types of services to monitor"""
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    CPU = "cpu"
    NETWORK = "network"


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    service_name: str
    service_type: ServiceType
    status: HealthStatus
    response_time_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class HealthChecker:
    """
    Comprehensive health checker for all system components.
    """

    def __init__(self):
        self.settings = get_settings()
        self._session: Optional[aiohttp.ClientSession] = None
        self._health_history: Dict[str, List[HealthCheckResult]] = {}
        self._circuit_breakers: Dict[str, Any] = {}

    async def initialize(self):
        """Initialize health checker"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        logger.info("Health checker initialized")

    async def cleanup(self):
        """Cleanup resources"""
        if self._session:
            await self._session.close()

    async def check_all_services(self) -> Dict[str, HealthCheckResult]:
        """
        Run health checks for all services.

        Returns:
            Dictionary of service name to health check result
        """
        checks = {
            "database": self.check_database,
            "redis": self.check_redis,
            "temporal": self.check_temporal,
            "llm_service": self.check_llm_service,
            "system_resources": self.check_system_resources,
            "disk_space": self.check_disk_space
        }

        results = {}

        # Run all checks concurrently
        tasks = [
            self._run_single_check(name, check_func)
            for name, check_func in checks.items()
        ]

        check_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, (name, _) in enumerate(checks.items()):
            result = check_results[i]
            if isinstance(result, Exception):
                results[name] = HealthCheckResult(
                    service_name=name,
                    service_type=ServiceType.EXTERNAL_API,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    error=str(result),
                    message=f"Health check failed: {result}"
                )
            else:
                results[name] = result

        # Store results in history
        for name, result in results.items():
            self._store_health_result(name, result)

        return results

    async def _run_single_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check with timing"""
        start_time = time.time()
        try:
            result = await check_func()
            result.response_time_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            return HealthCheckResult(
                service_name=name,
                service_type=ServiceType.EXTERNAL_API,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
                message=f"Health check exception: {e}"
            )

    async def check_database(self) -> HealthCheckResult:
        """Check database connectivity and performance"""
        try:
            # Import here to avoid circular imports
            import asyncpg

            config = self.settings.get_database_config()

            # Test connection
            conn = await asyncpg.connect(
                host=config.host,
                port=config.port,
                database=config.database,
                user=config.username,
                password=config.password,
                timeout=5
            )

            # Test simple query
            result = await conn.fetchval("SELECT 1")
            await conn.close()

            if result == 1:
                return HealthCheckResult(
                    service_name="database",
                    service_type=ServiceType.DATABASE,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=0,  # Will be set by caller
                    message="Database connection successful"
                )
            else:
                return HealthCheckResult(
                    service_name="database",
                    service_type=ServiceType.DATABASE,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message="Database query returned unexpected result"
                )

        except Exception as e:
            return HealthCheckResult(
                service_name="database",
                service_type=ServiceType.DATABASE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                error=str(e),
                message=f"Database connection failed: {e}"
            )

    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity and performance"""
        try:
            import redis.asyncio as redis

            config = self.settings.get_redis_config()

            # Create Redis connection
            redis_client = redis.Redis(
                host=config.host,
                port=config.port,
                db=config.database,
                password=config.password,
                socket_connect_timeout=5,
                socket_timeout=5
            )

            # Test ping
            pong = await redis_client.ping()
            await redis_client.close()

            if pong:
                return HealthCheckResult(
                    service_name="redis",
                    service_type=ServiceType.CACHE,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=0,
                    message="Redis connection successful"
                )
            else:
                return HealthCheckResult(
                    service_name="redis",
                    service_type=ServiceType.CACHE,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message="Redis ping failed"
                )

        except Exception as e:
            return HealthCheckResult(
                service_name="redis",
                service_type=ServiceType.CACHE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                error=str(e),
                message=f"Redis connection failed: {e}"
            )

    async def check_temporal(self) -> HealthCheckResult:
        """Check Temporal workflow service"""
        try:
            config = self.settings.get_temporal_config()

            if not self._session:
                await self.initialize()

            # Check Temporal health endpoint
            url = f"http://{config.host}:{config.port}/api/v1/namespaces/{config.namespace}"

            async with self._session.get(url) as response:
                if response.status == 200:
                    return HealthCheckResult(
                        service_name="temporal",
                        service_type=ServiceType.MESSAGE_QUEUE,
                        status=HealthStatus.HEALTHY,
                        response_time_ms=0,
                        message="Temporal service accessible"
                    )
                else:
                    return HealthCheckResult(
                        service_name="temporal",
                        service_type=ServiceType.MESSAGE_QUEUE,
                        status=HealthStatus.DEGRADED,
                        response_time_ms=0,
                        message=f"Temporal returned status {response.status}"
                    )

        except Exception as e:
            return HealthCheckResult(
                service_name="temporal",
                service_type=ServiceType.MESSAGE_QUEUE,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                error=str(e),
                message=f"Temporal health check failed: {e}"
            )

    async def check_llm_service(self) -> HealthCheckResult:
        """Check LLM service availability"""
        try:
            # Test AWS Bedrock connectivity
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError

            config = self.settings.get_llm_config()

            if not config.aws_access_key_id or not config.aws_secret_access_key:
                return HealthCheckResult(
                    service_name="llm_service",
                    service_type=ServiceType.EXTERNAL_API,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0,
                    message="AWS credentials not configured"
                )

            # Test Bedrock connection
            bedrock = boto3.client(
                'bedrock',
                region_name=config.aws_region,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )

            # List available models (light operation)
            models = bedrock.list_foundation_models()

            if models.get('modelSummaries'):
                return HealthCheckResult(
                    service_name="llm_service",
                    service_type=ServiceType.EXTERNAL_API,
                    status=HealthStatus.HEALTHY,
                    response_time_ms=0,
                    message="AWS Bedrock accessible",
                    details={"model_count": len(models['modelSummaries'])}
                )
            else:
                return HealthCheckResult(
                    service_name="llm_service",
                    service_type=ServiceType.EXTERNAL_API,
                    status=HealthStatus.DEGRADED,
                    response_time_ms=0,
                    message="AWS Bedrock accessible but no models available"
                )

        except NoCredentialsError:
            return HealthCheckResult(
                service_name="llm_service",
                service_type=ServiceType.EXTERNAL_API,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                message="AWS credentials invalid"
            )
        except ClientError as e:
            return HealthCheckResult(
                service_name="llm_service",
                service_type=ServiceType.EXTERNAL_API,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                error=str(e),
                message=f"AWS Bedrock error: {e}"
            )
        except Exception as e:
            return HealthCheckResult(
                service_name="llm_service",
                service_type=ServiceType.EXTERNAL_API,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                error=str(e),
                message=f"LLM service check failed: {e}"
            )

    async def check_system_resources(self) -> HealthCheckResult:
        """Check system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            # Determine overall status
            status = HealthStatus.HEALTHY

            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                status = HealthStatus.UNHEALTHY
            elif cpu_percent > 70 or memory_percent > 80 or disk_percent > 80:
                status = HealthStatus.DEGRADED

            return HealthCheckResult(
                service_name="system_resources",
                service_type=ServiceType.MEMORY,
                status=status,
                response_time_ms=0,
                message=f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%, Disk: {disk_percent:.1f}%",
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_percent,
                    "disk_percent": disk_percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_free_gb": disk.free / (1024**3)
                }
            )

        except Exception as e:
            return HealthCheckResult(
                service_name="system_resources",
                service_type=ServiceType.MEMORY,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                error=str(e),
                message=f"System resource check failed: {e}"
            )

    async def check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            used_percent = (disk_usage.used / disk_usage.total) * 100

            status = HealthStatus.HEALTHY
            if free_gb < 1:  # Less than 1GB free
                status = HealthStatus.UNHEALTHY
            elif free_gb < 5:  # Less than 5GB free
                status = HealthStatus.DEGRADED

            return HealthCheckResult(
                service_name="disk_space",
                service_type=ServiceType.FILE_SYSTEM,
                status=status,
                response_time_ms=0,
                message=f"Disk usage: {used_percent:.1f}%, Free: {free_gb:.1f}GB",
                details={
                    "free_gb": free_gb,
                    "used_percent": used_percent,
                    "total_gb": disk_usage.total / (1024**3)
                }
            )

        except Exception as e:
            return HealthCheckResult(
                service_name="disk_space",
                service_type=ServiceType.FILE_SYSTEM,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                error=str(e),
                message=f"Disk space check failed: {e}"
            )

    def get_system_metrics(self) -> SystemMetrics:
        """Get detailed system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            process_count = len(psutil.pids())

            # Load average (Unix systems)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                process_count=process_count,
                load_average=load_avg
            )

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0,
                load_average=[0.0, 0.0, 0.0]
            )

    def _store_health_result(self, service_name: str, result: HealthCheckResult):
        """Store health check result in history"""
        if service_name not in self._health_history:
            self._health_history[service_name] = []

        self._health_history[service_name].append(result)

        # Keep only last 100 results
        if len(self._health_history[service_name]) > 100:
            self._health_history[service_name] = self._health_history[service_name][-100:]

    def get_service_health_trends(self, service_name: str, hours: int = 24) -> Dict[str, Any]:
        """Get health trends for a service over specified hours"""
        if service_name not in self._health_history:
            return {"error": "No health data available"}

        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_results = [
            result for result in self._health_history[service_name]
            if result.timestamp > cutoff_time
        ]

        if not recent_results:
            return {"error": "No recent health data available"}

        # Calculate trends
        response_times = [r.response_time_ms for r in recent_results if r.response_time_ms > 0]
        healthy_count = len([r for r in recent_results if r.status == HealthStatus.HEALTHY])
        total_count = len(recent_results)

        return {
            "service_name": service_name,
            "period_hours": hours,
            "total_checks": total_count,
            "healthy_checks": healthy_count,
            "availability_percent": (healthy_count / total_count) * 100 if total_count > 0 else 0,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "recent_status": recent_results[-1].status if recent_results else HealthStatus.UNKNOWN,
            "recent_message": recent_results[-1].message if recent_results else ""
        }

    def get_overall_system_health(self) -> Dict[str, Any]:
        """Get overall system health summary"""
        all_results = {}

        # Get latest results for each service
        for service_name, results in self._health_history.items():
            if results:
                all_results[service_name] = results[-1]

        if not all_results:
            return {
                "overall_status": HealthStatus.UNKNOWN,
                "message": "No health data available"
            }

        # Determine overall status
        statuses = [result.status for result in all_results.values()]
        unhealthy_count = statuses.count(HealthStatus.UNHEALTHY)
        degraded_count = statuses.count(HealthStatus.DEGRADED)

        if unhealthy_count > 0:
            overall_status = HealthStatus.UNHEALTHY
            message = f"{unhealthy_count} service(s) unhealthy"
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
            message = f"{degraded_count} service(s) degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All services healthy"

        return {
            "overall_status": overall_status,
            "message": message,
            "services": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "response_time_ms": result.response_time_ms,
                    "last_check": result.timestamp.isoformat()
                }
                for name, result in all_results.items()
            },
            "summary": {
                "total_services": len(all_results),
                "healthy": statuses.count(HealthStatus.HEALTHY),
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "unknown": statuses.count(HealthStatus.UNKNOWN)
            }
        }


class CircuitBreakerManager:
    """
    Circuit breaker manager for external service protection.
    """

    def __init__(self):
        self._breakers: Dict[str, Any] = {}

    def get_breaker(
        self,
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        expected_exception: type = Exception
    ):
        """
        Get or create a circuit breaker for a service.

        Args:
            service_name: Name of the service
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery
            expected_exception: Exception type that triggers the circuit

        Returns:
            Circuit breaker decorator
        """
        if service_name not in self._breakers:
            self._breakers[service_name] = circuit(
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                expected_exception=expected_exception
            )

        return self._breakers[service_name]

    def get_breaker_status(self, service_name: str) -> Dict[str, Any]:
        """Get circuit breaker status"""
        if service_name not in self._breakers:
            return {"error": "Circuit breaker not found"}

        breaker = self._breakers[service_name]

        return {
            "service_name": service_name,
            "state": breaker.current_state,
            "failure_count": breaker.failure_count,
            "failure_threshold": breaker.failure_threshold,
            "recovery_timeout": breaker.recovery_timeout,
            "last_failure": breaker.last_failure_time
        }

    def reset_breaker(self, service_name: str) -> bool:
        """Reset a circuit breaker"""
        if service_name in self._breakers:
            self._breakers[service_name].reset()
            return True
        return False


# Global instances
health_checker = HealthChecker()
circuit_breaker_manager = CircuitBreakerManager()


async def get_health_checker() -> HealthChecker:
    """Get initialized health checker"""
    if health_checker._session is None:
        await health_checker.initialize()
    return health_checker


def get_circuit_breaker_manager() -> CircuitBreakerManager:
    """Get circuit breaker manager"""
    return circuit_breaker_manager
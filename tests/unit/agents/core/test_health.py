"""
Unit Tests for Health Check System
=================================

Tests for comprehensive health monitoring including:
- Service health checks
- Circuit breaker patterns
- Performance monitoring
- System metrics collection
- Health trends and analytics
"""

import pytest
import asyncio
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agents.core.health import (
    HealthChecker,
    CircuitBreakerManager,
    HealthCheckResult,
    SystemMetrics,
    HealthStatus,
    ServiceType,
    health_checker,
    circuit_breaker_manager,
    get_health_checker,
    get_circuit_breaker_manager
)


class TestHealthCheckResult:
    """Test HealthCheckResult functionality"""

    def test_health_check_result_creation(self):
        """Test creating health check result"""
        result = HealthCheckResult(
            service_name="test_service",
            service_type=ServiceType.DATABASE,
            status=HealthStatus.HEALTHY,
            response_time_ms=125.5,
            message="Service is healthy",
            details={"connections": 10}
        )

        assert result.service_name == "test_service"
        assert result.service_type == ServiceType.DATABASE
        assert result.status == HealthStatus.HEALTHY
        assert result.response_time_ms == 125.5
        assert result.message == "Service is healthy"
        assert result.details["connections"] == 10
        assert isinstance(result.timestamp, datetime)

    def test_health_check_result_with_error(self):
        """Test health check result with error"""
        result = HealthCheckResult(
            service_name="failing_service",
            service_type=ServiceType.EXTERNAL_API,
            status=HealthStatus.UNHEALTHY,
            response_time_ms=0,
            error="Connection timeout",
            message="Service unreachable"
        )

        assert result.status == HealthStatus.UNHEALTHY
        assert result.error == "Connection timeout"
        assert result.message == "Service unreachable"


class TestSystemMetrics:
    """Test SystemMetrics functionality"""

    def test_system_metrics_creation(self):
        """Test creating system metrics"""
        metrics = SystemMetrics(
            cpu_percent=45.2,
            memory_percent=67.8,
            disk_percent=82.1,
            network_io={"bytes_sent": 1024, "bytes_recv": 2048},
            process_count=156,
            load_average=[1.2, 1.5, 1.8]
        )

        assert metrics.cpu_percent == 45.2
        assert metrics.memory_percent == 67.8
        assert metrics.disk_percent == 82.1
        assert metrics.network_io["bytes_sent"] == 1024
        assert metrics.process_count == 156
        assert len(metrics.load_average) == 3
        assert isinstance(metrics.timestamp, datetime)


class TestHealthChecker:
    """Test HealthChecker functionality"""

    @pytest.fixture
    def health_checker_instance(self):
        """Create health checker instance for testing"""
        return HealthChecker()

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing"""
        settings = Mock()

        # Database config
        db_config = Mock()
        db_config.host = "localhost"
        db_config.port = 5432
        db_config.database = "test_db"
        db_config.username = "test_user"
        db_config.password = "test_pass"
        settings.get_database_config.return_value = db_config

        # Redis config
        redis_config = Mock()
        redis_config.host = "localhost"
        redis_config.port = 6379
        redis_config.database = 0
        redis_config.password = None
        settings.get_redis_config.return_value = redis_config

        # Temporal config
        temporal_config = Mock()
        temporal_config.host = "localhost"
        temporal_config.port = 7233
        temporal_config.namespace = "default"
        settings.get_temporal_config.return_value = temporal_config

        # LLM config
        llm_config = Mock()
        llm_config.aws_access_key_id = "test_key"
        llm_config.aws_secret_access_key = "test_secret"
        llm_config.aws_region = "us-east-1"
        settings.get_llm_config.return_value = llm_config

        return settings

    @pytest.mark.asyncio
    async def test_health_checker_initialization(self, health_checker_instance):
        """Test health checker initialization"""
        await health_checker_instance.initialize()

        assert health_checker_instance._session is not None

        await health_checker_instance.cleanup()

    @pytest.mark.asyncio
    async def test_health_checker_cleanup(self, health_checker_instance):
        """Test health checker cleanup"""
        await health_checker_instance.initialize()
        session = health_checker_instance._session

        await health_checker_instance.cleanup()

        # Session should be closed
        assert session.closed

    @pytest.mark.asyncio
    @patch('asyncpg.connect')
    async def test_check_database_success(self, mock_connect, health_checker_instance, mock_settings):
        """Test successful database health check"""
        health_checker_instance.settings = mock_settings

        # Mock successful connection
        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = 1
        mock_connect.return_value = mock_conn

        result = await health_checker_instance.check_database()

        assert result.service_name == "database"
        assert result.service_type == ServiceType.DATABASE
        assert result.status == HealthStatus.HEALTHY
        assert "Database connection successful" in result.message

    @pytest.mark.asyncio
    @patch('asyncpg.connect')
    async def test_check_database_failure(self, mock_connect, health_checker_instance, mock_settings):
        """Test database health check failure"""
        health_checker_instance.settings = mock_settings

        # Mock connection failure
        mock_connect.side_effect = Exception("Connection failed")

        result = await health_checker_instance.check_database()

        assert result.service_name == "database"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_check_redis_success(self, mock_redis_class, health_checker_instance, mock_settings):
        """Test successful Redis health check"""
        health_checker_instance.settings = mock_settings

        # Mock successful Redis connection
        mock_client = AsyncMock()
        mock_client.ping.return_value = True
        mock_redis_class.return_value = mock_client

        result = await health_checker_instance.check_redis()

        assert result.service_name == "redis"
        assert result.service_type == ServiceType.CACHE
        assert result.status == HealthStatus.HEALTHY
        assert "Redis connection successful" in result.message

    @pytest.mark.asyncio
    @patch('redis.asyncio.Redis')
    async def test_check_redis_failure(self, mock_redis_class, health_checker_instance, mock_settings):
        """Test Redis health check failure"""
        health_checker_instance.settings = mock_settings

        # Mock Redis connection failure
        mock_redis_class.side_effect = Exception("Redis unavailable")

        result = await health_checker_instance.check_redis()

        assert result.service_name == "redis"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Redis unavailable" in result.message

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_check_temporal_success(self, mock_get, health_checker_instance, mock_settings):
        """Test successful Temporal health check"""
        health_checker_instance.settings = mock_settings

        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200

        # Mock async context manager
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

        # Initialize session
        await health_checker_instance.initialize()

        result = await health_checker_instance.check_temporal()

        assert result.service_name == "temporal"
        assert result.service_type == ServiceType.MESSAGE_QUEUE
        assert result.status == HealthStatus.HEALTHY
        assert "Temporal service accessible" in result.message

        await health_checker_instance.cleanup()

    @pytest.mark.asyncio
    @patch('aiohttp.ClientSession.get')
    async def test_check_temporal_degraded(self, mock_get, health_checker_instance, mock_settings):
        """Test Temporal health check with degraded status"""
        health_checker_instance.settings = mock_settings

        # Mock HTTP response with error status
        mock_response = AsyncMock()
        mock_response.status = 503

        # Mock async context manager
        mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_get.return_value.__aexit__ = AsyncMock(return_value=None)

        # Initialize session
        await health_checker_instance.initialize()

        result = await health_checker_instance.check_temporal()

        assert result.service_name == "temporal"
        assert result.status == HealthStatus.DEGRADED
        assert "Temporal returned status 503" in result.message

        await health_checker_instance.cleanup()

    @pytest.mark.asyncio
    @patch('boto3.client')
    async def test_check_llm_service_success(self, mock_boto3_client, health_checker_instance, mock_settings):
        """Test successful LLM service health check"""
        health_checker_instance.settings = mock_settings

        # Mock successful Bedrock client
        mock_bedrock = Mock()
        mock_bedrock.list_foundation_models.return_value = {
            'modelSummaries': [{'modelId': 'claude-3-sonnet'}, {'modelId': 'claude-3-haiku'}]
        }
        mock_boto3_client.return_value = mock_bedrock

        result = await health_checker_instance.check_llm_service()

        assert result.service_name == "llm_service"
        assert result.service_type == ServiceType.EXTERNAL_API
        assert result.status == HealthStatus.HEALTHY
        assert "AWS Bedrock accessible" in result.message
        assert result.details["model_count"] == 2

    @pytest.mark.asyncio
    async def test_check_llm_service_no_credentials(self, health_checker_instance):
        """Test LLM service health check with no credentials"""
        # Mock settings with no credentials
        mock_settings = Mock()
        llm_config = Mock()
        llm_config.aws_access_key_id = None
        llm_config.aws_secret_access_key = None
        mock_settings.get_llm_config.return_value = llm_config
        health_checker_instance.settings = mock_settings

        result = await health_checker_instance.check_llm_service()

        assert result.service_name == "llm_service"
        assert result.status == HealthStatus.UNHEALTHY
        assert "AWS credentials not configured" in result.message

    @pytest.mark.asyncio
    @patch('agents.core.health.psutil')
    async def test_check_system_resources_healthy(self, mock_psutil, health_checker_instance):
        """Test system resources check with healthy status"""
        # Mock healthy system metrics
        mock_psutil.cpu_percent.return_value = 45.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 8 * (1024**3)  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.used = 50 * (1024**3)  # 50GB used
        mock_disk.total = 100 * (1024**3)  # 100GB total
        mock_disk.free = 50 * (1024**3)  # 50GB free
        mock_psutil.disk_usage.return_value = mock_disk

        result = await health_checker_instance.check_system_resources()

        assert result.service_name == "system_resources"
        assert result.service_type == ServiceType.MEMORY
        assert result.status == HealthStatus.HEALTHY
        assert "CPU: 45.0%" in result.message
        assert result.details["cpu_percent"] == 45.0
        assert result.details["memory_percent"] == 60.0

    @pytest.mark.asyncio
    @patch('agents.core.health.psutil')
    async def test_check_system_resources_degraded(self, mock_psutil, health_checker_instance):
        """Test system resources check with degraded status"""
        # Mock degraded system metrics
        mock_psutil.cpu_percent.return_value = 75.0  # Above 70%
        mock_memory = Mock()
        mock_memory.percent = 85.0  # Above 80%
        mock_memory.available = 2 * (1024**3)  # 2GB
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.used = 60 * (1024**3)
        mock_disk.total = 100 * (1024**3)
        mock_disk.free = 40 * (1024**3)
        mock_psutil.disk_usage.return_value = mock_disk

        result = await health_checker_instance.check_system_resources()

        assert result.service_name == "system_resources"
        assert result.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    @patch('agents.core.health.psutil')
    async def test_check_disk_space_low(self, mock_psutil, health_checker_instance):
        """Test disk space check with low space"""
        mock_disk = Mock()
        mock_disk.used = 99 * (1024**3)  # 99GB used
        mock_disk.total = 100 * (1024**3)  # 100GB total
        mock_disk.free = 0.5 * (1024**3)   # 0.5GB free (below 1GB threshold)
        mock_psutil.disk_usage.return_value = mock_disk

        result = await health_checker_instance.check_disk_space()

        assert result.service_name == "disk_space"
        assert result.service_type == ServiceType.FILE_SYSTEM
        assert result.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    @patch('agents.core.health.psutil')
    async def test_get_system_metrics(self, mock_psutil, health_checker_instance):
        """Test getting detailed system metrics"""
        # Mock psutil calls
        mock_psutil.cpu_percent.return_value = 50.0

        mock_memory = Mock()
        mock_memory.percent = 70.0
        mock_psutil.virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.used = 60 * (1024**3)
        mock_disk.total = 100 * (1024**3)
        mock_psutil.disk_usage.return_value = mock_disk

        mock_network = Mock()
        mock_network.bytes_sent = 1000000
        mock_network.bytes_recv = 2000000
        mock_network.packets_sent = 500
        mock_network.packets_recv = 800
        mock_psutil.net_io_counters.return_value = mock_network

        mock_psutil.pids.return_value = list(range(200))  # 200 processes
        mock_psutil.getloadavg.return_value = (1.2, 1.5, 1.8)

        metrics = health_checker_instance.get_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 50.0
        assert metrics.memory_percent == 70.0
        assert metrics.disk_percent == 60.0
        assert metrics.network_io["bytes_sent"] == 1000000
        assert metrics.process_count == 200
        assert metrics.load_average == [1.2, 1.5, 1.8]

    @pytest.mark.asyncio
    async def test_check_all_services(self, health_checker_instance):
        """Test checking all services"""
        # Mock individual check methods
        health_checker_instance.check_database = AsyncMock(return_value=HealthCheckResult(
            "database", ServiceType.DATABASE, HealthStatus.HEALTHY, 50.0, message="DB OK"
        ))
        health_checker_instance.check_redis = AsyncMock(return_value=HealthCheckResult(
            "redis", ServiceType.CACHE, HealthStatus.HEALTHY, 25.0, message="Redis OK"
        ))
        health_checker_instance.check_temporal = AsyncMock(return_value=HealthCheckResult(
            "temporal", ServiceType.MESSAGE_QUEUE, HealthStatus.DEGRADED, 100.0, message="Temporal slow"
        ))
        health_checker_instance.check_llm_service = AsyncMock(return_value=HealthCheckResult(
            "llm_service", ServiceType.EXTERNAL_API, HealthStatus.HEALTHY, 200.0, message="LLM OK"
        ))
        health_checker_instance.check_system_resources = AsyncMock(return_value=HealthCheckResult(
            "system_resources", ServiceType.MEMORY, HealthStatus.HEALTHY, 10.0, message="System OK"
        ))
        health_checker_instance.check_disk_space = AsyncMock(return_value=HealthCheckResult(
            "disk_space", ServiceType.FILE_SYSTEM, HealthStatus.HEALTHY, 5.0, message="Disk OK"
        ))

        results = await health_checker_instance.check_all_services()

        assert len(results) == 6
        assert "database" in results
        assert "redis" in results
        assert "temporal" in results
        assert "llm_service" in results
        assert "system_resources" in results
        assert "disk_space" in results

        assert results["database"].status == HealthStatus.HEALTHY
        assert results["temporal"].status == HealthStatus.DEGRADED

    def test_health_trends(self, health_checker_instance):
        """Test health trends calculation"""
        service_name = "test_service"

        # Add some mock health results
        now = datetime.now(timezone.utc)
        results = [
            HealthCheckResult(service_name, ServiceType.DATABASE, HealthStatus.HEALTHY, 100.0,
                            timestamp=now - timedelta(hours=2), message="OK"),
            HealthCheckResult(service_name, ServiceType.DATABASE, HealthStatus.HEALTHY, 150.0,
                            timestamp=now - timedelta(hours=1), message="OK"),
            HealthCheckResult(service_name, ServiceType.DATABASE, HealthStatus.DEGRADED, 300.0,
                            timestamp=now - timedelta(minutes=30), message="Slow"),
        ]

        for result in results:
            health_checker_instance._store_health_result(service_name, result)

        trends = health_checker_instance.get_service_health_trends(service_name, hours=24)

        assert trends["service_name"] == service_name
        assert trends["total_checks"] == 3
        assert trends["healthy_checks"] == 2
        assert trends["availability_percent"] == (2/3) * 100
        assert trends["avg_response_time_ms"] == (100 + 150 + 300) / 3
        assert trends["recent_status"] == HealthStatus.DEGRADED

    def test_overall_system_health(self, health_checker_instance):
        """Test overall system health calculation"""
        # Add mock results for multiple services
        services_data = [
            ("database", HealthStatus.HEALTHY),
            ("redis", HealthStatus.HEALTHY),
            ("temporal", HealthStatus.DEGRADED),
            ("llm_service", HealthStatus.UNHEALTHY),
        ]

        for service_name, status in services_data:
            result = HealthCheckResult(
                service_name, ServiceType.DATABASE, status, 100.0,
                message=f"{service_name} status"
            )
            health_checker_instance._store_health_result(service_name, result)

        overall_health = health_checker_instance.get_overall_system_health()

        assert overall_health["overall_status"] == HealthStatus.UNHEALTHY  # Has unhealthy service
        assert "1 service(s) unhealthy" in overall_health["message"]
        assert overall_health["summary"]["total_services"] == 4
        assert overall_health["summary"]["healthy"] == 2
        assert overall_health["summary"]["degraded"] == 1
        assert overall_health["summary"]["unhealthy"] == 1


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager functionality"""

    @pytest.fixture
    def cb_manager(self):
        """Create circuit breaker manager for testing"""
        return CircuitBreakerManager()

    def test_get_breaker(self, cb_manager):
        """Test getting a circuit breaker"""
        service_name = "test_service"
        breaker = cb_manager.get_breaker(service_name)

        assert service_name in cb_manager._breakers
        assert breaker is not None

    def test_get_existing_breaker(self, cb_manager):
        """Test getting an existing circuit breaker"""
        service_name = "test_service"
        breaker1 = cb_manager.get_breaker(service_name)
        breaker2 = cb_manager.get_breaker(service_name)

        # Should return the same breaker instance
        assert breaker1 is breaker2

    def test_breaker_with_custom_params(self, cb_manager):
        """Test creating breaker with custom parameters"""
        service_name = "custom_service"
        breaker = cb_manager.get_breaker(
            service_name,
            failure_threshold=10,
            recovery_timeout=60,
            expected_exception=ConnectionError
        )

        assert service_name in cb_manager._breakers
        assert breaker is not None

    def test_get_breaker_status_not_found(self, cb_manager):
        """Test getting status for non-existent breaker"""
        status = cb_manager.get_breaker_status("non_existent")

        assert "error" in status
        assert status["error"] == "Circuit breaker not found"

    def test_reset_breaker_success(self, cb_manager):
        """Test resetting an existing breaker"""
        service_name = "test_service"
        cb_manager.get_breaker(service_name)  # Create breaker

        result = cb_manager.reset_breaker(service_name)

        assert result is True

    def test_reset_breaker_not_found(self, cb_manager):
        """Test resetting non-existent breaker"""
        result = cb_manager.reset_breaker("non_existent")

        assert result is False


class TestGlobalInstances:
    """Test global health check instances"""

    @pytest.mark.asyncio
    async def test_get_health_checker(self):
        """Test getting global health checker"""
        checker = await get_health_checker()

        assert isinstance(checker, HealthChecker)
        assert checker._session is not None

    def test_get_circuit_breaker_manager(self):
        """Test getting global circuit breaker manager"""
        manager = get_circuit_breaker_manager()

        assert isinstance(manager, CircuitBreakerManager)

    def test_global_instances_singleton(self):
        """Test that global instances are singletons"""
        manager1 = get_circuit_breaker_manager()
        manager2 = get_circuit_breaker_manager()

        assert manager1 is manager2


class TestHealthCheckIntegration:
    """Integration tests for health check system"""

    @pytest.mark.asyncio
    async def test_health_check_timing(self):
        """Test that health checks properly measure response time"""
        checker = HealthChecker()

        # Mock a slow check
        async def slow_check():
            await asyncio.sleep(0.1)  # 100ms delay
            return HealthCheckResult(
                "slow_service", ServiceType.DATABASE, HealthStatus.HEALTHY, 0
            )

        result = await checker._run_single_check("slow_service", slow_check)

        # Should have measured the time (approximately 100ms)
        assert result.response_time_ms >= 100
        assert result.response_time_ms < 200  # Some tolerance

    @pytest.mark.asyncio
    async def test_health_check_exception_handling(self):
        """Test health check exception handling"""
        checker = HealthChecker()

        # Mock a failing check
        async def failing_check():
            raise ConnectionError("Service unavailable")

        result = await checker._run_single_check("failing_service", failing_check)

        assert result.status == HealthStatus.UNHEALTHY
        assert "Service unavailable" in result.error
        assert result.response_time_ms >= 0

    def test_health_history_management(self):
        """Test health history storage and cleanup"""
        checker = HealthChecker()
        service_name = "test_service"

        # Add more than 100 results
        for i in range(150):
            result = HealthCheckResult(
                service_name, ServiceType.DATABASE, HealthStatus.HEALTHY, 100.0
            )
            checker._store_health_result(service_name, result)

        # Should keep only last 100 results
        assert len(checker._health_history[service_name]) == 100

    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test that health checks can run concurrently"""
        checker = HealthChecker()

        # Mock multiple quick checks
        checker.check_database = AsyncMock(return_value=HealthCheckResult(
            "database", ServiceType.DATABASE, HealthStatus.HEALTHY, 50.0
        ))
        checker.check_redis = AsyncMock(return_value=HealthCheckResult(
            "redis", ServiceType.CACHE, HealthStatus.HEALTHY, 25.0
        ))

        start_time = time.time()
        results = await checker.check_all_services()
        end_time = time.time()

        # Should complete quickly since they run concurrently
        assert (end_time - start_time) < 2.0  # Less than 2 seconds (more lenient for test environment)
        assert len(results) >= 2


if __name__ == "__main__":
    pytest.main([__file__])
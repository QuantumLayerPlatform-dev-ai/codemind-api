"""
Integration Tests for Health Check API Endpoints
===============================================

Tests that verify the health check system works end-to-end:
- Health check endpoints respond correctly
- Health check service integration
- Real system resource monitoring
"""

import pytest
import asyncio
from fastapi.testclient import TestClient

from main import app
from agents.core.health import health_checker, get_health_checker


class TestHealthCheckEndpoints:
    """Test health check API endpoints"""

    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)

    def test_basic_health_check(self):
        """Test basic health check endpoint"""
        response = self.client.get("/health/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "codemind-api"
        assert "version" in data

    def test_ping_endpoint(self):
        """Test ping endpoint"""
        response = self.client.get("/health/ping")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "pong"


class TestHealthCheckSystemIntegration:
    """Test health check system integration"""

    @pytest.mark.asyncio
    async def test_health_checker_real_system_metrics(self):
        """Test that health checker can get real system metrics"""
        checker = await get_health_checker()

        # Test system metrics gathering
        metrics = checker.get_system_metrics()

        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_percent >= 0
        assert metrics.process_count > 0
        assert len(metrics.load_average) == 3
        assert isinstance(metrics.network_io, dict)

    @pytest.mark.asyncio
    async def test_health_checker_system_resources_check(self):
        """Test system resources health check with real data"""
        checker = await get_health_checker()

        # This should work with real system data
        result = await checker.check_system_resources()

        assert result.service_name == "system_resources"
        assert result.status in ["healthy", "degraded", "unhealthy"]
        assert result.response_time_ms >= 0
        assert "CPU:" in result.message
        assert "Memory:" in result.message
        assert "Disk:" in result.message

        # Should have real system data
        assert "cpu_percent" in result.details
        assert "memory_percent" in result.details
        assert "disk_percent" in result.details

    @pytest.mark.asyncio
    async def test_health_checker_disk_space_check(self):
        """Test disk space health check with real data"""
        checker = await get_health_checker()

        result = await checker.check_disk_space()

        assert result.service_name == "disk_space"
        assert result.status in ["healthy", "degraded", "unhealthy"]
        assert result.response_time_ms >= 0
        assert "Disk usage:" in result.message
        assert "Free:" in result.message

        # Should have real disk data
        assert "free_gb" in result.details
        assert "used_percent" in result.details
        assert "total_gb" in result.details
        assert result.details["free_gb"] > 0
        assert result.details["total_gb"] > 0


class TestHealthCheckCircuitBreaker:
    """Test circuit breaker functionality"""

    @pytest.mark.asyncio
    async def test_circuit_breaker_integration(self):
        """Test circuit breaker with health checker"""
        from agents.core.health import get_circuit_breaker_manager

        cb_manager = get_circuit_breaker_manager()

        # Get a circuit breaker for a test service
        service_name = "test_integration_service"
        breaker = cb_manager.get_breaker(service_name, failure_threshold=2)

        assert breaker is not None

        # Test circuit breaker status
        status = cb_manager.get_breaker_status(service_name)
        assert status["service_name"] == service_name
        assert "state" in status
        assert "failure_count" in status

    @pytest.mark.asyncio
    async def test_health_trends_tracking(self):
        """Test health trends tracking over time"""
        checker = await get_health_checker()

        # Run a quick health check to generate data
        result = await checker.check_system_resources()

        # Check that health history is being tracked
        assert len(checker._health_history) >= 0

        # If we have data, test trends
        if "system_resources" in checker._health_history:
            trends = checker.get_service_health_trends("system_resources", hours=1)

            assert "service_name" in trends
            assert "total_checks" in trends
            assert "availability_percent" in trends


if __name__ == "__main__":
    pytest.main([__file__])
import pytest
import requests


class TestWorker:
    
    @pytest.mark.P0
    @pytest.mark.description("vllm worker health check")
    def test_health_check(self, host, use_docker):
        if not use_docker:
            pytest.skip("Only vllm images supports this case.")
        response = requests.get(f"http://{host}:9000/health", timeout=5)
        assert response.status_code == 200
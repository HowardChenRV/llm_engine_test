import pytest
import requests
from ...utils.logger import logger


class TestCountToken:
    headers = { "Content-Type": "application/json" }
    
    @pytest.mark.P1
    @pytest.mark.description("vllm count prompt token")
    def test_extra_apis_count_token(self, host, port, model):
        url = f"http://{host}:{port}/count_token"
        data = {
            "model": model,
            "prompt": "Who are you?"
        }
        response = requests.post(url, headers=self.headers, json=data, timeout=5)
        assert response.status_code == 200
        logger.debug(response.text)
        
        response_json = response.json()
        assert response_json["model"] == model
        assert response_json["prompt"] == "Who are you?"
        assert response_json["token_count"] > 0
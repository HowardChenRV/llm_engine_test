import pytest
import requests
from ...utils.logger import logger


class TestBestSamplingParams:
    headers = { "Content-Type": "application/json" }
    
    @pytest.mark.P1
    @pytest.mark.description("vllm get model best sampling params")
    def test_extra_apis_get_model_best_sampling_params(self, host, port, model):
        url = f"http://{host}:{port}/get_model_best_sampling_params"
        data = {
            "model": model
        }
        response = requests.post(url, headers=self.headers, json=data, timeout=5)
        assert response.status_code == 200
        logger.debug(response.text)
        
        response_json = response.json()
        assert response_json["model"] == model
        sampling_params = response_json["sampling_params"]
        assert 0 <= sampling_params["top_p"] <= 1 or \
            sampling_params["top_p"] == None
        assert sampling_params["top_k"] == -1 or \
            sampling_params["top_k"] >= 1 or \
            sampling_params["top_k"] == None
        assert 0 <= sampling_params["temperature"] <= 2 or \
            sampling_params["temperature"] == None
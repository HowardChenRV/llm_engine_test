import pytest
import requests
from ...utils.logger import logger


class TestModelCard:
    headers = { "Content-Type": "application/json" }

    @pytest.mark.skip(reason="No longer supported")
    @pytest.mark.P2
    @pytest.mark.description("vllm get model list")
    def test_models(self, host, port, model):
        url = f"http://{host}:{port}/v1/models"
        response = requests.get(url, headers=self.headers, timeout=5)
        assert response.status_code == 200
        logger.debug(response.text)
        
        response_json = response.json()
        assert response_json["object"] == "list"
        models = response_json["data"]
        assert isinstance(models, list) and len(models) >= 1
        assert models[0]["object"] == "model"
        assert models[0]["id"] == model

    @pytest.mark.P2
    @pytest.mark.description("vllm get model details")
    def test_extra_apis_get_model_details(self, host, port, model):
        url = f"http://{host}:{port}/get_model_details"
        data = {
            "model": model
        }
        response = requests.post(url, headers=self.headers, json=data, timeout=5)
        assert response.status_code == 200
        logger.debug(response.text)
        
        response_json = response.json()
        assert response_json["model"] == model
        assert response_json["context_length"] > 0
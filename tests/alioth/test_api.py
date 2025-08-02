import sys
import traceback
import pytest
import requests
import json
from transformers import AutoConfig
from ...utils.logger import logger


class TestWorker:
    headers = { "Content-Type": "application/json" }
    
    @pytest.mark.P0
    @pytest.mark.description("worker测试: health")
    def test_worker_health(self, host, port):
        response = requests.get(f"http://{host}:{port}/health", headers=self.headers)
        assert response.status_code == 200
    
    @pytest.mark.P2
    @pytest.mark.description("worker测试: engine version")
    def test_engine_version(self, host, port):
        response = requests.get(f"http://{host}:{port}/version", headers=self.headers)
        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data["version"]) > 0
        
        
class TestModelCard:
    headers = { "Content-Type": "application/json" }
    
    @pytest.mark.P2
    @pytest.mark.description("Model Card测试: id、root、object、permission、max_model_len")
    def test_model_card(self, host, port, model):
        try:
            # 加载模型配置
            hf_model_config = AutoConfig.from_pretrained(model)
            # 请求        
            response = requests.get(f"http://{host}:{port}/v1/models", headers=self.headers)
            assert response.status_code == 200
            data = json.loads(response.content)["data"][0]
            assert data["id"] == model
            assert data["root"] == model
            # assert data["owned_by"] == "sglang"
            assert data["object"] == "model"
            assert len(data["permission"]) > 0
            assert data["max_model_len"] == hf_model_config.max_position_embeddings
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
        

class TestTokenizer:
    headers = { "Content-Type": "application/json" }
    
    @pytest.mark.P2
    @pytest.mark.description("Tokenizer测试: tokenize")
    def test_tokenize(self, host, port, model):
        data = {
            "model": model,
            "prompt": "你是谁？",
            "add_special_tokens": True
        }
        try:
            # 加载模型配置
            hf_model_config = AutoConfig.from_pretrained(model)
            # 请求        
            response = requests.post(f"http://{host}:{port}/tokenize", headers=self.headers, json=data)
            assert response.status_code == 200
            response_json = response.json()
            assert len(response_json["tokens"]) > 0
            assert response_json["count"] == len(response_json["tokens"])
            assert response_json["max_model_len"] == hf_model_config.max_position_embeddings
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            
    @pytest.mark.P2
    @pytest.mark.description("Tokenizer测试: detokenize")
    def test_detokenize(self, host, port, model):
        data = {
            "model": model,
            "tokens": [666, 666, 666],
        }
        try:
            response = requests.post(f"http://{host}:{port}/detokenize", headers=self.headers, json=data)
            assert response.status_code == 200
            response_json = response.json()
            assert len(response_json["prompt"]) > 0
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
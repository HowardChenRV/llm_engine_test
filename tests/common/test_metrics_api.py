import sys
import traceback
import requests
import pytest
from ...utils.logger import logger
from ...utils.prometheus_tool import parse_prometheus_metrics, REQUIRED_METRICS


class TestMetrics:
    
    @pytest.mark.P1
    @pytest.mark.description("Metrics test: prometheus data format validation, required metrics data validation")
    def test_metrics(self, client, host, port, model, engine, use_docker):
        if use_docker:
            url = f"http://{host}:20000/metrics"    # vllm image starts metrics on a separate listening port
        else:
            url = f"http://{host}:{port}/metrics"

        if engine not in ["sglang", "llm", "vllm"]:
            pytest.skip(f"{engine} not support metrics")
                
        request_content = {
            "model": model,
            "max_tokens": 1024,
            "messages": [
                { "role": "user", "content": "Please give me the full text of the first chapter of 《Pride and Prejudice》."  }
            ]
        }
        
        try:
            # First make a request to generate data for metrics
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            
            # Request metrics
            response = requests.get(url, timeout=5)
            assert response.status_code == 200
            content = response.content.decode("utf-8")
            logger.debug(f"Metrics content = \n{content}")
            
            assert content != "" , "metrics is None"
            metrics = parse_prometheus_metrics(content)
            # Validate required metrics
            required_metrics = REQUIRED_METRICS[engine] + REQUIRED_METRICS["common"]
            for metric in required_metrics:
                metric_name, metric_type = metric
                # Validate metric existence
                assert metric_name in metrics
                assert metric_type == metrics[metric_name]["type"]
                assert metrics[metric_name]["help"] is not None
                # Validate metric values
                metric_values = metrics[metric_name]["values"]
                assert metric_values is not None
                
                for metric_value in metric_values:
                    
                    # if metric_name.startswith(engine) and metric_name != f"{engine}:cache_config_info":
                    if "engine" in metric_value and metric_name != "cache_config_info":
                        assert metric_value["engine"] == engine
                        
                        if use_docker:
                            assert metric_value["model_name"] == f"/app/model/{model}"
                        else:
                            assert metric_value["model_name"] == model
                        if metric_type == "histogram":
                            assert metric_value["label"] in ["bucket", "sum", "count"]
                            if metric_value["label"] == "bucket":
                                assert metric_value["le"] is not None and metric_value["le"] != ""
                    assert isinstance(metric_value["value"], float) and metric_value["value"] >= 0.0
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
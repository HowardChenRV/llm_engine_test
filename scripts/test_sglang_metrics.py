import sys
import os

sys.path.append("/Users/howardchen/Dev/QA/llm_engine_test")

from utils.prometheus_tool import parse_prometheus_metrics


# Open and read file content
with open('scripts/metrics.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# Print or use the read content
print(content)
print("-" * 100)

# Parse Prometheus format monitoring metrics
metrics = parse_prometheus_metrics(content)

# print(metrics)
use_docker = False
model = "deepseek-r1"
engine = "sglang"

# Validate required metrics
# required_metrics = REQUIRED_METRICS[engine] + REQUIRED_METRICS["common"]
for metric in metrics:
    
    # Validate metric values
    metric_values = metrics[metric]["values"]
    metric_type = metrics[metric]["type"]
    metric_help = metrics[metric]["help"]
    assert metric_values is not None
    assert metric_type is not None
    assert metric_help is not None
        
    for metric_value in metric_values:
        print(metric, metric_value)
        
        # if metric_name.startswith(engine) and metric_name != f"{engine}:cache_config_info":
        # if "engine" in metric_value and metric_name != "cache_config_info":
        #     assert metric_value["engine"] == engine
            
        #     if use_docker:
        #         assert metric_value["model_name"] == f"/app/model/{model}"
        #     else:
        #         assert metric_value["model_name"] == model
        #     if metric_type == "histogram":
        #         assert metric_value["label"] in ["bucket", "sum", "count"]
        #         if metric_value["label"] == "bucket":
        #             assert metric_value["le"] is not None and metric_value["le"] != ""
        # assert isinstance(metric_value["value"], float) and metric_value["value"] >= 0.0
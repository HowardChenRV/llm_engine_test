import sys
import os

sys.path.append("/Users/howardchen/Dev/QA/llm_engine_test")

from utils.prometheus_tool import parse_prometheus_metrics


# 打开并读取文件内容
with open('scripts/metrics.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 打印或使用读取的内容
print(content)
print("-" * 100)

# 解析普罗米修斯格式监控指标
metrics = parse_prometheus_metrics(content)

# print(metrics)
use_docker = False
model = "deepseek-r1"
engine = "sglang"

# 校验必须metrics
# required_metrics = REQUIRED_METRICS[engine] + REQUIRED_METRICS["common"]
for metric in metrics:
    
    # 校验metric值
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
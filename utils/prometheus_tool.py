import re
from typing import Dict, Any
# utils
from .logger import logger


REQUIRED_METRICS = {
    "sglang": [
        # Cache Config Information
        ("cache_config_info", "gauge"),
        # Scheduler State
        ("num_requests_running", "gauge"),
        ("num_requests_waiting", "gauge"),
        # KV Cache Usage in %
        ("gpu_cache_usage_perc", "gauge"),
        # Iteration stats
        ("num_blocking_steps_total", "counter"),
        ("num_non_blocking_steps_total", "counter"),
        ("num_preemptions_total", "counter"),
        ("prompt_tokens_total", "counter"),
        ("generation_tokens_total", "counter"),
        ("time_to_first_token_seconds", "histogram"),
        ("time_per_output_token_seconds", "histogram"),
        ("time_schedule", "histogram"),
        ("time_prepare_input", "histogram"),
        ("time_prepare_sample", "histogram"),
        ("time_forward", "histogram"),
        ("time_sample", "histogram"),
        ("time_process_model_outputs", "histogram"),
        ("forward_batch_size_iter", "gauge"),
        # Latency
        ("e2e_request_latency_seconds", "histogram"),
        # Metadata
        ("request_prompt_tokens", "histogram"),
        ("request_generation_tokens", "histogram"),
        ("request_params_best_of", "histogram"),
        ("request_params_n", "histogram"),
        ("request_success_total", "counter"),
        
        # # Cache Config Information
        # ("sglang:cache_config_info", "gauge"),
        # # Scheduler State
        # ("sglang:num_requests_running", "gauge"),
        # ("sglang:num_requests_waiting", "gauge"),
        # # KV Cache Usage in %
        # ("sglang:gpu_cache_usage_perc", "gauge"),
        # # Iteration stats
        # ("sglang:num_blocking_steps_total", "counter"),
        # ("sglang:num_non_blocking_steps_total", "counter"),
        # ("sglang:num_preemptions_total", "counter"),
        # ("sglang:prompt_tokens_total", "counter"),
        # ("sglang:generation_tokens_total", "counter"),
        # ("sglang:time_to_first_token_seconds", "histogram"),
        # ("sglang:time_per_output_token_seconds", "histogram"),
        # ("sglang:time_schedule", "histogram"),
        # ("sglang:time_prepare_input", "histogram"),
        # ("sglang:time_prepare_sample", "histogram"),
        # ("sglang:time_forward", "histogram"),
        # ("sglang:time_sample", "histogram"),
        # ("sglang:time_process_model_outputs", "histogram"),
        # ("sglang:forward_batch_size_iter", "gauge"),
        # # Latency
        # ("sglang:e2e_request_latency_seconds", "histogram"),
        # # Metadata
        # ("sglang:request_prompt_tokens", "histogram"),
        # ("sglang:request_generation_tokens", "histogram"),
        # ("sglang:request_params_best_of", "histogram"),
        # ("sglang:request_params_n", "histogram"),
        # ("sglang:request_success_total", "counter"),
        # Deprecated in favor of sglang:prompt_tokens_total
        # ("sglang:avg_prompt_throughput_toks_per_s", "gauge"),
        # Deprecated in favor of sglang:generation_tokens_total
        # ("sglang:avg_generation_throughput_toks_per_s", "gauge"),
        # Prefix Caching
        # ("sglang:n_blocks_allocated", "counter"),
        # ("sglang:number_blocks_hitted_gpu", "counter"),
        # ("sglang:number_blocks_hitted_cpu", "counter"),
        # ("sglang:num_swap_in", "counter"),
        # ("sglang:num_swap_out", "counter"),
    ],
    "vllm": [
        # Cache Config Information
        ("cache_config_info", "gauge"),
        # 调度指标
        ("num_requests_running", "gauge"),
        ("num_requests_swapped", "gauge"),
        ("num_requests_waiting", "gauge"),
        # cache利用率
        ("gpu_cache_usage_perc", "gauge"),
        ("cpu_cache_usage_perc", "gauge"),
        # 输入输出token数
        ("prompt_tokens_total", "counter"),
        ("generation_tokens_total", "counter"),
        # 延时
        ("time_to_first_token_seconds", "histogram"),
        ("time_per_output_token_seconds", "histogram"),
        ("e2e_request_latency_seconds", "histogram"),
        # 吞吐
        ("avg_prompt_throughput_toks_per_s", "gauge"),
        ("avg_generation_throughput_toks_per_s", "gauge"),
        # # Cache Config Information
        # ("vllm:cache_config_info", "gauge"),
        # # 调度指标
        # ("vllm:num_requests_running", "gauge"),
        # ("vllm:num_requests_swapped", "gauge"),
        # ("vllm:num_requests_waiting", "gauge"),
        # # cache利用率
        # ("vllm:gpu_cache_usage_perc", "gauge"),
        # ("vllm:cpu_cache_usage_perc", "gauge"),
        # # 输入输出token数
        # ("vllm:prompt_tokens_total", "counter"),
        # ("vllm:generation_tokens_total", "counter"),
        # # 延时
        # ("vllm:time_to_first_token_seconds", "histogram"),
        # ("vllm:time_per_output_token_seconds", "histogram"),
        # ("vllm:e2e_request_latency_seconds", "histogram"),
        # # 吞吐
        # ("vllm:avg_prompt_throughput_toks_per_s", "gauge"),
        # ("vllm:avg_generation_throughput_toks_per_s", "gauge"),
    ],
    "common": [
        # python指标
        ("python_gc_objects_collected_total", "counter"),
        ("python_gc_objects_uncollectable_total", "counter"),
        ("python_gc_collections_total", "counter"),
        ("python_info", "gauge"),
        # 系统指标
        ("process_virtual_memory_bytes", "gauge"),
        ("process_resident_memory_bytes", "gauge"),
        ("process_start_time_seconds", "gauge"),
        ("process_cpu_seconds_total", "counter"),
        ("process_open_fds", "gauge"),
        ("process_max_fds", "gauge")
    ]
}


# 解析普罗米修斯格式监控指标，解析失败报错
def parse_prometheus_metrics(metrics_text: str) -> Dict[str, Any]:
    logger.debug(metrics_text)
    metrics = {}
    lines = metrics_text.strip().split('\n')
    current_metric = None

    try:
        for line in lines:
            if line.startswith("# HELP"):
                parts = line.split(' ', 3)
                current_metric = parts[2]
                metrics[current_metric] = {"help": parts[3], "values": []}
            elif line.startswith("# TYPE"):
                parts = line.split(' ', 3)
                if current_metric:
                    metrics[current_metric]["type"] = parts[3]
            else:
                match = re.match(r'([\w:]+)\{?(.*?)\}? (\d+\.?\d*)', line)
                if match:
                    metric_name, labels, value = match.groups()
                    # 处理histogram类型的bucket
                    if metric_name.endswith("_bucket"):
                        metric_name = metric_name.removesuffix("_bucket")
                        label_dict = {"label": "bucket"}
                    elif metric_name.endswith("_count"):
                        metric_name = metric_name.removesuffix("_count")
                        label_dict = {"label": "count"}
                    elif metric_name.endswith("_sum"):
                        metric_name = metric_name.removesuffix("_sum")
                        label_dict = {"label": "sum"}
                    else:
                        label_dict = {}
                    # 处理label
                    if labels:
                        label_pairs = labels.split(',')
                        for pair in label_pairs:
                            key, val = pair.split('=')
                            label_dict[key] = val.strip('"')
                    label_dict["value"] = float(value)
                    # 处理value
                    if "values" not in metrics[metric_name]:
                        metrics[metric_name]["values"] = []
                    metrics[metric_name]["values"].append(label_dict)
    except Exception as e:
        logger.error(f"Parse Prometheus metrics error: {metrics_text}")
        raise AssertionError(f"Parse Prometheus metrics error: {e}") 
                
    return metrics
    
    
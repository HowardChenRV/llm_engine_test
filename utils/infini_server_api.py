import time
import asyncio
from aiohttp import ClientSession

import requests
import json
from transformers import AutoTokenizer

# utils
from .logger import logger
from .prometheus_tool import parse_prometheus_metrics


class llmServerApi:
    
    # TIMEOUT = 120               # 请求超时时间, 120s
    TIMEOUT = 20

    @classmethod
    def ping_health(cls, host: str, port: str, model_name: str = None) -> bool:
        url = f"http://{host}:{port}/health"
        logger.info(f"Try to ping {url}, with {model_name}")
        try:
            response = requests.get(url, timeout=cls.TIMEOUT)
            assert response.status_code == 200, f"Ping {url} fail: HTTP status code is not 200."
            logger.info(f"Ping {url} finish.")
            return True
        except Exception as e:
            logger.error(f"Ping {url} failed: {e}")
            return False      
        
    
    @classmethod
    def ping_generate(cls, host: str, port: str, model_name: str = None) -> bool:
        url = f"http://{host}:{port}/generate"
        logger.info(f"Try to ping {url}, with {model_name}")

        headers = { 'Content-Type': 'application/json' }
        data = {
            "prompt": "The biggest animal in the world is",
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 10,
            "ignore_eos": False,
            "stream": False,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=cls.TIMEOUT)
            assert response.status_code == 200, f"Ping {url} fail: HTTP status code is not 200."
            data_json = response.json()
            logger.info(f"Ping serving response: {data_json}")
            text_list = data_json.get("text", [])
            out_len = data_json.get("out_len", None)
            assert out_len == 10
            assert len(text_list) > 0
            text = text_list[0]
            assert text is not None and text != ""
            assert "blue whale" in text.lower(), f"Ping {url} fail: prompt answer error, text = {text}"
            logger.info(f"Ping {url} finish.")
            return True
        except Exception as e:
            logger.error(f"Ping {url} failed: {e}")
            return False


    @classmethod
    def ping_generate_stream(cls, host: str, port: str, model_name: str = None) -> bool:
        url = f"http://{host}:{port}/generate"
        logger.info(f"Try to ping {url} by stream, with {model_name}")

        headers = { 'Content-Type': 'application/json' }
        data = {
            "prompt": "The biggest animal in the world is",
            "n": 1,
            "best_of": 1,
            "use_beam_search": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 128,
            "ignore_eos": False,
            "stream": True,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=cls.TIMEOUT, stream=True)
            assert response.status_code == 200, f"Ping {url} by stream fail: HTTP status code is not 200."
            
            # Initialize an empty list to collect text data
            text = ""

            # Iterate over response stream lines
            iter_count = 1
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    logger.debug(f"Stream Iterate: {line}")
                    try:
                        data_json = json.loads(line)
                        text_list = data_json.get("text", [])
                        out_len = data_json.get("out_len", None)
                        assert out_len == iter_count
                        assert len(text_list) > 0
                        text = text_list[0]
                        assert text is not None and text != ""
                        iter_count += 1
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON line: {line}, error: {e}")
                        continue
            
            assert "blue whale" in text.lower(), f"Ping {url} fail: prompt answer error, text = {text}"
            logger.info(f"Ping {url} by stream finish.")
            return True
        except Exception as e:
            logger.error(f"Ping {url} by stream failed: {e}")
            return False
 
 
    @classmethod
    def ping_metrics(cls, host: str, port: str, model_name: str = None) -> bool:
        url = f"http://{host}:{port}/metrics/"
        logger.info(f"Try to ping {url}, with {model_name}")
        try:
            response = requests.get(url, timeout=cls.TIMEOUT)
            assert response.status_code == 200, f"Ping {url} fail: HTTP status code is not 200."
            content = response.content.decode('utf-8')
            logger.debug(f"Metrics content = {content}")

            required_metrics = [
                # Config Information
                ("llm:cache_config_info", "gauge"),
                # 调度指标
                ("llm:num_requests_running", "gauge"),
                ("llm:num_requests_swapped", "gauge"),
                ("llm:num_requests_waiting", "gauge"),
                # cache利用率
                ("llm:gpu_cache_usage_perc", "gauge"),
                ("llm:cpu_cache_usage_perc", "gauge"),
                # 输入输出token数
                ("llm:prompt_tokens_total", "counter"),
                ("llm:generation_tokens_total", "counter"),
                # 延时
                ("llm:time_to_first_token_seconds", "histogram"),
                ("llm:time_per_output_token_seconds", "histogram"),
                ("llm:e2e_request_latency_seconds", "histogram"),
                # 吞吐
                ("llm:avg_prompt_throughput_toks_per_s", "gauge"),
                ("llm:avg_generation_throughput_toks_per_s", "gauge"),
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
            
            # 解析prometheus的metrics格式, 解析失败会抛出
            metrics = parse_prometheus_metrics(content)
            # 校验metrics
            for metric in required_metrics:
                metric_name, metric_type = metric
                # 校验metric存在性
                assert metric_name in metrics
                assert metric_type == metrics[metric_name]["type"]
                assert metrics[metric_name]["help"] is not None
                # 校验metric值
                metric_values = metrics[metric_name]["values"]
                assert metric_values is not None
                for metric_value in metric_values:
                    if metric_name.startswith("llm") and metric_name != "llm:cache_config_info":
                        if model_name:
                            assert metric_value["model_name"] == model_name
                        if metric_type == "histogram":
                            assert metric_value["label"] in ["bucket", "sum", "count"]
                            if metric_value["label"] == "bucket":
                                assert metric_value["le"] is not None and metric_value["le"] != ""
                    assert isinstance(metric_value["value"], float) and metric_value["value"] >= 0.0
            # 校验成功
            logger.info(f"Ping {url} finish.")
            return True
        except Exception as e:
            logger.error(f"Ping {url} failed: {e}")
            return False

    # 正确性测试
    @classmethod
    def post_generate(cls, host: str, port: int, query: str):
        url = f"http://{host}:{port}/generate"
        headers = {'Content-Type': 'application/json'}
        data = {
            "prompt": query,
            "temperature": 0.0,
            "max_tokens": 50,
            "ignore_eos": True,
        }
        logger.info(f"Try to ping {url} data {data}")

        try:
            s = requests.Session()
            response = s.post(url, headers=headers, json=data, timeout=cls.TIMEOUT)
            assert response.status_code == 200, f"Ping {url} fail: HTTP status code is not 200."
            data_json = response.json()
            s.close()
            time.sleep(10)
            return data_json
        except Exception as e:
            logger.error(f"Post {url} failed: {e}")
            return False

    @classmethod
    async def post_correct_async(cls, host: str, port: int, idx: int,  query: str):
        logger.info(f'题目{idx} 启动时间: {time.time()}')
        url = f"http://{host}:{port}/generate"
        headers = {'Content-Type': 'application/json'}
        data = {
            "prompt": query,
            "temperature": 0.0,
            "max_tokens": 50,
            "ignore_eos": True,
        }
        logger.info(f"Try to ping {url} data {data}")
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=cls.TIMEOUT) as response:
                res = await response.json()
                return idx, res
    

class OpenAiApi:
    pass

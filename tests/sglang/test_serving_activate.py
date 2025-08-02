import json
import subprocess
import multiprocessing
import time
import concurrent.futures
import os

import pytest
import requests
import asyncio
from transformers import AutoTokenizer

# utils
from ...utils.logger import logger
from ...utils.nvidia_tool import set_cuda_visible_devices
from ...utils.system_tool import is_port_listening, terminate_all_child_processes
from ...utils.llm_server_api import llmServerApi
from ...utils.llm_command import llmServingCommand


class TestServingActivate:
    
    SERVING_HOST = "0.0.0.0"    # Service host address
    LISTEN_PORT = 8000          # apiserver listening port
    TIMEOUT = 300               # serving startup timeout, 5min


    # Monitor serving startup
    @classmethod
    def serving_activate_monitor(cls, serving_process, result_container):
        base_url = f"http://{cls.SERVING_HOST}:{cls.LISTEN_PORT}"
        start_time = time.time()
        
        error_list = [
            "address already in use",
            "Address already in use",
            "out of memory"
        ]
        
        ping_result = multiprocessing.Manager().dict()
        ping_result["ping_success"] = False
        ping_result["model_name"] = result_container["model_name"]
        ping_process = multiprocessing.Process(target=cls.ping_serving_port, args=(ping_result,))

        while time.time() - start_time < cls.TIMEOUT:
            # Monitor standard output
            output = serving_process.stdout.readline().strip()
            # print(output, flush=True)
            if base_url in output:
                logger.info(f"Found log {base_url}.")
                if is_port_listening(cls.SERVING_HOST, cls.LISTEN_PORT):
                    duration = time.time() - start_time
                    logger.info(f"{base_url} listening, duration is {duration} seconds.")
                    result_container["activate_result"] = True
                    result_container["duration"] = duration

                    # Start another process to execute ping_serving_port
                    ping_process.start()
                    ping_process.join()
                    
                    result_container["ping_success"] = ping_result["ping_success"]
                    return
            else:
                # Startup failed
                for error_info in error_list:
                    if error_info in output:
                        logger.error(error_info)
                        result_container["activate_result"] = False
                        return
                
        logger.error(f"{base_url} is not listening within {cls.TIMEOUT} seconds.")
        result_container["activate_result"] = False
        return


    # Interface testing
    @classmethod
    def ping_serving_port(cls, result_container):
        def run_task(name, func, host, port, model_name):
            try:
                return (name, func(host, port, model_name))
            except Exception as e:
                logger.error(f"Task {name} failed: {e}")
                return (name, False)
            
        tasks = {
            "ping_health": llmServerApi.ping_health,
            "ping_generate": llmServerApi.ping_generate,
            "ping_generate_stream": llmServerApi.ping_generate_stream,
            "ping_metrics": llmServerApi.ping_metrics
        }
        
        # Start 4 threads to ping different interfaces
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_task, name, func, cls.SERVING_HOST, cls.LISTEN_PORT, result_container["model_name"]): name for name, func in tasks.items()}
            
            results = {}
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                results[name] = result
        
        for name, result in results.items():
            if result:
                logger.info(f"Task {name} succeeded.")
            else:
                logger.error(f"Task {name} failed.")
                result_container["ping_success"] = False
                return
        result_container["ping_success"] = True
        return


    # Test Serving startup
    @pytest.mark.P0
    @pytest.mark.parametrize("model", [
        {"tp": 1, "model_path": "/share/datasets/public_models/Llama-2-7b-chat-hf"},
        {"tp": 1, "model_path": "/share/datasets/public_models/Meta-Llama-3-8B-Instruct"},
        {"tp": 1, "model_path": "/share/datasets/public_models/Qwen_Qwen1.5-7B-Chat"},
        {"tp": 1, "model_path": "/share/datasets/public_models/Qwen_Qwen2-7B-Instruct"},
        {"tp": 2, "model_path": "/share/datasets/public_models/Llama-2-7b-chat-hf"},
        {"tp": 2, "model_path": "/share/datasets/public_models/Meta-Llama-3-8B-Instruct"},
        {"tp": 2, "model_path": "/share/datasets/public_models/Qwen_Qwen1.5-7B-Chat"},
        {"tp": 2, "model_path": "/share/datasets/public_models/Qwen_Qwen2-7B-Instruct"},
    ])
    def test_serving_activate(self, model):
        # Process test parameters
        tp = model["tp"]
        model_path = model["model_path"]
        logger.info(f"Test '{model_path}' with tp='{tp}'.")
        
        # Set CUDA_VISIBLE_DEVICES
        set_cuda_visible_devices(gpu_num=tp)
        
        # Set llm log level
        os.environ["INFI_LOG_LEVEL"] = "3"


        # Startup command
        llm_serving_command = llmServingCommand(
            model_path = model_path,
            tp = tp
        )
        command = llm_serving_command.get_default_command()

        # Container for storing monitoring results
        result_container = multiprocessing.Manager().dict()
        result_container["model_name"] = model_path
        
        try:
            # Start Serving
            process = subprocess.Popen(
                command,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                bufsize = 1
            )
            
            # Wait for Serving to start
            self.serving_activate_monitor(process, result_container)
            
            # Validate: serving startup successful, inference successful
            activate_result = result_container.get("activate_result", False)
            ping_success = result_container.get("ping_success", False)
            duration = result_container.get("duration", 0)
            logger.info(result_container)
            assert activate_result and ping_success and duration < self.TIMEOUT
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            pytest.fail(f"An unexpected error occurred: {e}")
        finally:
            # Close Serving
            if process:
                logger.info(f"Killing main PID: {process.pid}")
                # List and terminate child processes
                terminate_all_child_processes(process.pid)
                time.sleep(5)
                # Terminate main process
                process.kill()
                process.wait()

    
    # Test ray startup
    @pytest.mark.P1
    @pytest.mark.parametrize("model", [
        {"tp": 2, "model_path": "/share/datasets/public_models/Meta-Llama-3-8B-Instruct"},
        {"tp": 2, "model_path": "/share/datasets/public_models/Qwen_Qwen2-7B-Instruct"},
    ])
    def test_serving_ray_backend(self, model):
        # Process test parameters
        tp = model["tp"]
        model_path = model["model_path"]
        logger.info(f"Test '{model_path}' with tp='{tp}'.")
        
        # Set CUDA_VISIBLE_DEVICES
        set_cuda_visible_devices(gpu_num=tp)
        
        # Set llm log level
        os.environ["INFI_LOG_LEVEL"] = "3"
        # Set ray startup related parameters
        os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "0"

        # Startup command
        llm_serving_command = llmServingCommand(
            model_path = model_path,
            tp = tp
        )
        command = llm_serving_command.get_different_backend_command("ray")

        # Container for storing monitoring results
        result_container = multiprocessing.Manager().dict()
        result_container["model_name"] = model_path
        
        try:
            # Start Serving
            process = subprocess.Popen(
                command,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
                text = True,
                bufsize = 1
            )
            
            # Wait for Serving to start
            self.serving_activate_monitor(process, result_container)
            
            # Validate: serving startup successful, inference successful
            activate_result = result_container.get("activate_result", False)
            ping_success = result_container.get("ping_success", False)
            duration = result_container.get("duration", 0)
            logger.info(result_container)
            assert activate_result and ping_success and duration < self.TIMEOUT
            
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            pytest.fail(f"An unexpected error occurred: {e}")
        finally:
            # Close Serving
            if process:
                logger.info(f"Killing main PID: {process.pid}")
                # List and terminate child processes
                terminate_all_child_processes(process.pid)
                time.sleep(5)
                # Terminate main process
                process.kill()
                process.wait()

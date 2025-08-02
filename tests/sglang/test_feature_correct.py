import subprocess
import multiprocessing
import time
import datetime
import concurrent.futures
import os
import pandas as pd
import requests
import asyncio
import pytest
# utils
from ...utils.logger import logger
from ...utils.nvidia_tool import get_available_gpus
from ...utils.system_tool import is_port_listening, terminate_all_child_processes
from ...utils.llm_server_api import llmServerApi
from ...utils.llm_command import llmServingCommand
from ...utils.constant import *
from ...utils.hf_transformers import run_correct_hf


class TestFeatureCorrect:
    
    SERVING_HOST = "0.0.0.0"    # Service host address
    LISTEN_PORT = 8000          # apiserver listening port
    TIMEOUT = 1200               # serving startup timeout, 20min

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
        ping_process = multiprocessing.Process(target=cls.ping_serving_port, args=(ping_result,))

        while time.time() - start_time < cls.TIMEOUT:
            # Monitor standard output
            output = serving_process.stdout.readline().strip()
            print(output, flush=True)
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
        def run_task(name, func, host, port):
            try:
                return (name, func(host, port))
            except Exception as e:
                logger.error(f"Task {name} failed: {e}")
                return (name, False)
            
        tasks = {
            "ping_health": llmServerApi.ping_health,
            "ping_generate": llmServerApi.ping_generate,
            "ping_metrics": llmServerApi.ping_metrics,
            # "ping_generate_stream": llmServerApi.ping_generate_stream
        }
        
        # Start 4 threads to ping different interfaces
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(run_task, name, func, cls.SERVING_HOST, cls.LISTEN_PORT): name for name, func in tasks.items()}
            
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

    # Test feature functionality
    @pytest.mark.parametrize("feature_name", ["prefix_caching", "spec_decoding", "peak_memory_predict", "dynamic_strategy"])
    def test_serving_feature(self, feature_name):
        feature_cfg = Feature_config[feature_name]
        tp = feature_cfg["tp"]
        model_path = feature_cfg["model_path"]
        logger.info(f"Test feature_name '{feature_name}' with tp='{tp}' model_path='{model_path}'.")
        
        # Get available GPUs
        available_gpus = get_available_gpus()
        assert len(available_gpus) >= tp, "Not enough GPUs."
        
        # 设置 CUDA_VISIBLE_DEVICES
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus[:tp]))
        logger.info("export CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"])
        
        # 设置llm日志等级
        os.environ["INFI_LOG_LEVEL"] = "3"

        # Container for storing monitoring results
        res_container_base = multiprocessing.Manager().dict()
        res_container_target = multiprocessing.Manager().dict()

        # Get startup command
        llm_serving_command = llmServingCommand(model_path=model_path, tp=tp)
        command_base = llm_serving_command.get_default_command()
        command_target = llm_serving_command.get_feature_command(feature_name)

        # Container for storing model call results
        res = [{"idx": x + 1} for x in range(len(Question_10))]
        
        # Call huggingface transformers as baseline correctness, only add single GPU for now
        if tp == 1:
            run_correct_hf(model_path, Question_10, res)
        else:
            for x in res:
                x["hf"] = ""

        task_tuple = (("base", command_base, res_container_base), ("target", command_target, res_container_target))
        for name, command, result_container in task_tuple:
            try:
                # Start Serving
                logger.info(f"start command: {command}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                time.sleep(30 * tp)

                # Wait for Serving to start
                self.serving_activate_monitor(process, result_container)
                
                # Validate: serving startup successful
                activate_result = result_container.get("activate_result", False)
                ping_success = result_container.get("ping_success", False)
                duration = result_container.get("duration", 0)
                logger.info(result_container)
                assert activate_result and ping_success and duration < self.TIMEOUT

                async def call_correct():
                    task_list = []
                    for i in range(len(Question_10)):
                        task = asyncio.create_task(llmServerApi.post_correct_async(self.SERVING_HOST, self.LISTEN_PORT, i, Question_10[i]))
                        task_list.append(task)
                    done, pending = await asyncio.wait(task_list, timeout=None)
                    # Get execution results
                    for done_task in done:
                        idx, response = done_task.result()
                        text = response["text"][0][len(Question_10[idx]):]
                        res[idx][name] = text
                        logger.info(f"question {idx} finish, text: {text}")

                start_time = time.time()
                loop = asyncio.get_event_loop()
                loop.run_until_complete(call_correct())
                logger.info(f"Total time: {time.time() - start_time}")

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

        # Compare both results, save results
        df = pd.DataFrame(res)
        df["score_hf"] = df.apply(lambda row: 1 if row["base"] == row["hf"] else 0, axis=1)
        df["score"] = df.apply(lambda row: 1 if row["base"] == row["target"] else 0, axis=1)
        correct_hf = df["score_hf"].sum() == len(Question_10)
        correct = df["score"].sum() == len(Question_10)

        output_dir = "/share/datasets/tmp_share/chenyonghua/test_result/llm/feature"
        today = datetime.date.today().strftime('%Y_%m_%d')
        output_name = f"Auto_test_feature_{feature_name}_correct_{today}.csv"
        output = os.path.join(output_dir, output_name)
        df.to_csv(output, index=False)
        logger.info(f"feature {feature_name} correct test finish, feature is {correct}, hf is {correct_hf}")

    # Test feature functionality - Call an existing service
    def bak_test_serving_feature_as_client(self):
        # Complete inference, get answers
        res = [{"idx": x + 1} for x in range(len(Question_10))]
        for idx, question in enumerate(Question_10):
            logger.info(f"idx {idx} question {question} start")
            response = llmServerApi.post_generate(self.SERVING_HOST, self.LISTEN_PORT, question)
            logger.info(f"post_generate response: {response}")
            if not response:
                continue
            text = response["text"][0][len(question):]
            res[idx]["base"] = text
            logger.info(f"idx {idx} question {question} finish text {text}")

        df = pd.DataFrame(res)
        print(df)

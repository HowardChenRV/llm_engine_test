import os
import sys
import time
import yaml
import pytest
import argparse
import subprocess
from datetime import datetime
from utils.testsuite import TestsuiteResult
from utils.logger import logger
from utils.nvidia_tool import set_cuda_visible_devices
from utils.llm_command import sglangServingCommand
from utils.system_tool import terminate_all_child_processes
from utils.docker_tool import get_run_kwargs, run_container, remove_container, get_container_status
from prettytable import PrettyTable


def test_one_model(model, engine, use_docker=False, image=""):
    # start test
    if use_docker:
        result = test_one_model_with_docker(model, engine, image)
    else:
        result = test_one_model_with_sdk(model, engine)
        
    assert result, "测试失败, 测试结果为空"        
    # show report table
    report_table = PrettyTable()
    report_table.field_names = ["model_id", "tp", "passed", "failed", "xfailed", "skipped", "total"]
    report_table.add_row([model["id"], model["tp"], result.passed, result.failed, result.xfailed, result.skipped, result.total])
    print(report_table)

    # show failed testcases
    if result.exitcode != 0:
        print(f"\n{result.failed} failed testcases:")
        idx = 0
        for report in result.reports:
            if report.outcome == "failed":
                idx += 1
                print(f"{idx}: {report.nodeid}")


def test_one_model_with_docker(model, engine, image) -> TestsuiteResult:
    TIMEOUT = 900
    run_kwargs = get_run_kwargs(model, image)

    try:
        # 1. run container
        run_container(model["id"], **run_kwargs)

        # 2. wait for container ready
        start_time = time.time()
        while True:
            status = get_container_status(model["id"])
            logger.info(f"Listening container status = {status}")
            if status["active"] and status["connected"]:
                break
            
            # 检查是否超时
            elapsed_time = time.time() - start_time
            if elapsed_time > TIMEOUT:
                logger.error("Wait for container timeout!")
                remove_container(model["id"])
                sys.exit(1)
            
            time.sleep(5)

        # 3. 执行测试
        logger.info("start to test ...")
        result = TestsuiteResult()
        current_datetime = datetime.now()
        date_str = current_datetime.strftime("%Y-%m-%d")
        time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        report_dir = f"reports/{date_str}"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"test_report_{image.replace(':', '-')}_{model['id']}_tp{model['tp']}_{time_str}.html")
        
        pytest.main([
            "tests/vllm/",
            "tests/common/",
            "-vvs",
            "--model", model["id"],
            "--model-path", model["path"],
            "--max-model-len", str(model["max_model_len"]),
            "--engine", engine,
            "--use-docker",
            f"--html={report_file}",
            "--self-contained-html"
        ], plugins=[result])
        
        return result
    except Exception as e:
        logger.error(e)
    # 4. remove container
    finally:
        remove_container(model["id"])
    
    return None


def test_one_model_with_sdk(model, engine) -> TestsuiteResult:
    assert engine in ["sglang"], "当前引擎不支持auto test"
    # 1. 处理测试参数
    tp = model["tp"]
    model_path = model["path"]
    blocks = model.get("blocks", 2000)
    max_model_len = model.get("max_model_len", 4096)
    logger.info(f"Test '{model_path}' with tp='{tp}'.")
    
    # 设置 CUDA_VISIBLE_DEVICES
    set_cuda_visible_devices(gpu_num=tp)
    
    # 设置llm日志等级
    if engine == "sglang":
        os.environ["sglang_LOG_LEVEL"] = "3"
        additional_arg = { "--num-gpu-blocks-override": str(blocks), "--max-num-batched-tokens": str(max_model_len) }
        serving_command = sglangServingCommand(
            model_path=model_path,
            tp=tp,
            **additional_arg
        )
        
    # 初始化启动命令
    command = serving_command.get_default_command()
    logger.info(command)

    try:
        # 2. 启动Serving
        process = subprocess.Popen(
            command,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text = True,
            bufsize = 1
        )
        logger.info(f"serving process id: {process.pid}")
        
        # 2. 等待Serving启动
        assert serving_command.serving_activate_monitor(process), "Serving启动失败"
        
        # 3. 执行测试
        logger.info("start to test ...")
        result = TestsuiteResult()
        current_datetime = datetime.now()
        date_str = current_datetime.strftime("%Y-%m-%d")
        time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        report_dir = f"reports/{date_str}"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, f"test_report_{model['id']}_tp{model['tp']}_{time_str}.html")
        
        pytest.main([
            f"tests/{engine}/",
            "tests/common/",
            "-vs",
            "--model", model_path,
            "--model-path", model_path,
            "--engine", engine,
            "--max-model-len", str(model["max_model_len"]),
            "--host", serving_command.SERVING_HOST,
            "--port", str(serving_command.LISTEN_PORT),
            f"--html={report_file}",
            "--self-contained-html"
        ], plugins=[result])
        
        return result
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        pytest.fail(f"An unexpected error occurred: {e}")
    finally:
        # 关闭Serving
        if process:
            logger.info(f"Try to kill main PID: {process.pid}")
            # 列出并终止子进程
            terminate_all_child_processes(process.pid)
            time.sleep(5)
            # 终止主进程
            process.kill()
            process.wait()
            
    return None


def test_all_models(models, engine, model_id="", use_docker=False, image=""):
    for idx, model in enumerate(models):
        if model_id != "":   # 传了model_id就只测试一个模型，否则全测
            if model_id == model["id"]:
                logger.info(f"\n({idx+1}/{len(models)}) test {model['id']} ...")
                test_one_model(model, engine, use_docker, image)
                return
        else:
            logger.info(f"\n({idx+1}/{len(models)}) test {model['id']} ...")
            test_one_model(model, engine, use_docker, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto test for all models")
    parser.add_argument("--models-yaml", type=str, default="conf/vllm_sglang_nvidia_models.yaml", help="models file")
    parser.add_argument("--model-id", type=str, default="", help="ID of the model (eg. llama-3-8b-instruct), if you just want to test one model. (optional)")
    parser.add_argument("--image-tag", type=str, default="", help="If use docker, please input docker image tag")
    args = parser.parse_args()

    # read models yaml
    with open(args.models_yaml, errors="ignore") as f:
        models_yaml = yaml.safe_load(f)
        engine = models_yaml["engine"]
        device = models_yaml.get("device", "nvidia")
        use_docker = models_yaml.get("use_docker", False)
        parentdir = models_yaml["parentdir"]
        models = models_yaml["models"]
        
        for model in models:
            if "path" not in model:
                model["path"] = os.path.join(parentdir, model["subdir"])
            del model["subdir"]

    if use_docker:
        assert args.image_tag, f"测试vllm镜像部署必须传入 '--image-tag'"
        image = f"{engine}-{device}:{args.image_tag}"
        test_all_models(models, engine, args.model_id, use_docker, image)
    else:
        test_all_models(models, engine, args.model_id)
    

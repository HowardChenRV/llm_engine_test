import time
from typing import List, Optional
# utils
from .logger import logger
from .constant import *
from .system_tool import is_port_listening


class sglangServingCommand:
    SERVING_HOST = "0.0.0.0"    # 服务主机地址
    LISTEN_PORT = 8000        # apiserver监听端口
    TIMEOUT = 600               # serving启动超时时间，10min
    
    # 默认启动命令
    @classmethod
    def build_default_command(
        cls, 
        model_path: str, 
        tp: int=1, 
        quantization: Optional[str] = None,
        **kwargs) -> List[str]:
        
        command = [
            "python", "-m", "sglang_llm.entrypoints.openai.api_server",
            "--model", model_path,
            "-tp", str(tp),
            "--trust-remote-code"
        ]
        
        # 拼接量化方法
        if quantization:
            command = command + ["-q", quantization]
            
        # 拼接剩余全部参数
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"{key}")
            else:
                command.extend([f"{key}", str(value)])
        
        return command

    def __init__(
        self,
        model_path: str, 
        tp: int = 1,
        quantization: Optional[str] = None,
        **kwargs
    ):
        """
        根据给定的 tp 和 model_path 构建命令。

        :param tp: tensor parallelism (1, 2, or 4)
        :param model_path: 模型路径
        :param quantization: 量化方法，支持 awq
        :return: 命令列表
        """
        self.model_path = model_path
        
        assert tp in [1, 2, 4]
        self.tp = tp
        
        if quantization:
            assert quantization in ["awq"]
        self.quantization = quantization
            
        self.command = self.build_default_command(
            model_path = self.model_path,
            tp = self.tp,
            quantization = self.quantization,
            **kwargs
        )
        logger.info(f"llm Serving default command: {self.command}")
        
        
    def get_default_command(self):
        return self.command
    
    # TODO: 追加prefix caching的适配

    
    #监测serving启动
    def serving_activate_monitor(self, serving_process) -> bool:
        base_url = f"http://{self.SERVING_HOST}:{self.LISTEN_PORT}"
        start_time = time.time()

        while time.time() - start_time < self.TIMEOUT:
            # 检查进程是否已退出
            if serving_process.poll() is not None:
                if serving_process.stderr:
                    error_output = serving_process.stderr.read().strip()
                    logger.error(f"Serving process has exited unexpectedly. Error: {error_output}")
                else:
                    logger.error(f"Serving process has exited unexpectedly, but no error output was captured.")
                return False
            # 监测标准输出
            output = serving_process.stdout.readline().strip()
            print(output, flush=True)
            # if base_url in output:
            #     logger.info(f"Found log {base_url}.")
            # 监测端口监听
            if is_port_listening(self.SERVING_HOST, self.LISTEN_PORT):
                duration = time.time() - start_time
                logger.info(f"{base_url} listening, duration is {duration} seconds.")
                return True
                
        logger.error(f"{base_url} is not listening within {self.TIMEOUT} seconds.")
        return False


class llmServingCommand:
    
    # 默认启动命令
    @classmethod
    def build_default_command(
        cls, 
        model_path: str, 
        tp: int=1, 
        quantization: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs) -> List[str]:
        
        base_command = [
            "python", "-m", "llm.serving.api_server",
            "--model", model_path,
            "--tokenizer", tokenizer_path,
            "-tp", str(tp),
            "--trust-remote-code"
        ]
        
        # 拼接多卡
        if tp == 4:
            endpoints = ["localhost:8080", "localhost:8081", "localhost:8082", "localhost:8083"]
        elif tp == 2:
            endpoints = ["localhost:8080", "localhost:8081"]
        else:
            endpoints = []
        command = base_command + (["--endpoints"] + endpoints if endpoints else [])
        
        # 拼接量化方法
        if quantization:
            command = command + ["-q", quantization]
            
        # 拼接剩余全部参数
        for key, value in kwargs.items():
            if isinstance(value, bool):
                if value:
                    command.append(f"{key}")
            else:
                command.extend([f"{key}", str(value)])
        
        return command

    def __init__(
        self,
        model_path: str, 
        tp: int = 1,
        quantization: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs
    ):
        """
        根据给定的 tp 和 model_path 构建命令。

        :param tp: tensor parallelism (1, 2, or 4)
        :param model_path: 模型路径
        :param quantization: 量化方法，支持 awq
        :param tokenizer_path: 不传默认模型路径
        :return: 命令列表
        """
        self.model_path = model_path
        
        assert tp in [1, 2, 4]
        self.tp = tp
        
        if quantization:
            assert quantization in ["awq"]
        self.quantization = quantization
        
        if not tokenizer_path:
            self.tokenizer_path = model_path
            
        self.command = self.build_default_command(
            model_path = self.model_path,
            tp = self.tp,
            quantization = self.quantization,
            tokenizer_path = self.tokenizer_path,
            **kwargs
        )
        logger.info(f"llm Serving default command: {self.command}")
        
    def get_default_command(self):
        return self.command
    
    # 其他feature指令封装
    def get_feature_command(self, feature_name):
        cmd_suffix = Feature_config[feature_name]["cmd_suffix"]
        cmd = self.command + cmd_suffix
        return cmd

    def get_different_backend_command(self, backend: str):
        assert backend in ["ray", "rpyc"]
        return self.command + ["--backend", backend]


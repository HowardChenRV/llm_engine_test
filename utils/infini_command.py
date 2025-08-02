import time
from typing import List, Optional
# utils
from .logger import logger
from .constant import *
from .system_tool import is_port_listening


class sglangServingCommand:
    SERVING_HOST = "0.0.0.0"    # Service host address
    LISTEN_PORT = 8000        # apiserver listening port
    TIMEOUT = 600               # serving startup timeout, 10min
    
    # Default startup command
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
        
        # Concatenate quantization method
        if quantization:
            command = command + ["-q", quantization]
            
        # Concatenate all remaining parameters
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
        Build a command based on the given tp and model_path.

        :param tp: tensor parallelism (1, 2, or 4)
        :param model_path: Model path
        :param quantization: Quantization method, supports awq
        :return: Command list
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
    
    # TODO: Add prefix caching adaptation

    
    # Monitor serving startup
    def serving_activate_monitor(self, serving_process) -> bool:
        base_url = f"http://{self.SERVING_HOST}:{self.LISTEN_PORT}"
        start_time = time.time()

        while time.time() - start_time < self.TIMEOUT:
            # Check if the process has exited
            if serving_process.poll() is not None:
                if serving_process.stderr:
                    error_output = serving_process.stderr.read().strip()
                    logger.error(f"Serving process has exited unexpectedly. Error: {error_output}")
                else:
                    logger.error(f"Serving process has exited unexpectedly, but no error output was captured.")
                return False
            # Monitor standard output
            output = serving_process.stdout.readline().strip()
            print(output, flush=True)
            # if base_url in output:
            #     logger.info(f"Found log {base_url}.")
            # Monitor port listening
            if is_port_listening(self.SERVING_HOST, self.LISTEN_PORT):
                duration = time.time() - start_time
                logger.info(f"{base_url} listening, duration is {duration} seconds.")
                return True
                
        logger.error(f"{base_url} is not listening within {self.TIMEOUT} seconds.")
        return False


class llmServingCommand:
    
    # Default startup command
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
        
        # Concatenate multi-GPU
        if tp == 4:
            endpoints = ["localhost:8080", "localhost:8081", "localhost:8082", "localhost:8083"]
        elif tp == 2:
            endpoints = ["localhost:8080", "localhost:8081"]
        else:
            endpoints = []
        command = base_command + (["--endpoints"] + endpoints if endpoints else [])
        
        # Concatenate quantization method
        if quantization:
            command = command + ["-q", quantization]
            
        # Concatenate all remaining parameters
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
        Build a command based on the given tp and model_path.

        :param tp: tensor parallelism (1, 2, or 4)
        :param model_path: Model path
        :param quantization: Quantization method, supports awq
        :param tokenizer_path: If not provided, defaults to model path
        :return: Command list
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
    
    # Other feature command encapsulation
    def get_feature_command(self, feature_name):
        cmd_suffix = Feature_config[feature_name]["cmd_suffix"]
        cmd = self.command + cmd_suffix
        return cmd

    def get_different_backend_command(self, backend: str):
        assert backend in ["ray", "rpyc"]
        return self.command + ["--backend", backend]


import os
import pynvml
from typing import List
# utils
from .logger import logger


def get_available_gpus() -> List[int]:
    # Initialize NVML
    pynvml.nvmlInit()

    available_gpus = []
    device_count = pynvml.nvmlDeviceGetCount()

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # If GPU memory usage is 0, add to available GPU list
        if memory_info.used / (1024 * 1024 * 1024) < 1:
            available_gpus.append(i)

    # Shutdown NVML
    pynvml.nvmlShutdown()

    return available_gpus


def set_cuda_visible_devices(gpu_num: int):
    # Get available GPUs
    available_gpus = get_available_gpus()
    assert len(available_gpus) >= gpu_num, "Not enough GPUs."
    
    # Set CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus[:gpu_num]))
    logger.info("export CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"])
import socket
import psutil
# utils
from .logger import logger


# 监测端口监听
def is_port_listening(host, port) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    

# 关闭指定进程下所有子进程
def terminate_all_child_processes(parent_pid):
    try:
        # 获取主进程对象
        parent_process = psutil.Process(parent_pid)
        
        # 获取所有子进程，包括递归子进程
        child_processes = parent_process.children(recursive=True)
        
        for child in child_processes:
            logger.info(f"Attempting to terminate Child PID: {child.pid}, cmdline: {child.cmdline()}")
            child.kill()
    except psutil.NoSuchProcess:
        logger.error("The process does not exist.")
    except psutil.AccessDenied:
        logger.error("Access denied to process.")
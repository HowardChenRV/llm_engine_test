import socket
import psutil
# utils
from .logger import logger


# Monitor port listening
def is_port_listening(host, port) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
    

# Terminate all child processes under a specified process
def terminate_all_child_processes(parent_pid):
    try:
        # Get the parent process object
        parent_process = psutil.Process(parent_pid)
        
        # Get all child processes, including recursive child processes
        child_processes = parent_process.children(recursive=True)
        
        for child in child_processes:
            logger.info(f"Attempting to terminate Child PID: {child.pid}, cmdline: {child.cmdline()}")
            child.kill()
    except psutil.NoSuchProcess:
        logger.error("The process does not exist.")
    except psutil.AccessDenied:
        logger.error("Access denied to process.")
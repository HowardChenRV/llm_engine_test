import os
import sys
import docker
import requests
from .nvidia_tool import set_cuda_visible_devices
from .logger import logger

# docker harbor: https://harbor/projects/16/repositories
Engines = ["vllm", "sglang"]
Chips = ["nvidia", "amd", "iluvatar", "enflame", "metax", "biren", "moorethreads"]


def get_run_kwargs(model, image):
    # extract engine and chip from image ({engine}-{chip}:{tag})
    engine, chip = image.split("-")[0], image.split("-")[1].split(":")[0]

    assert os.path.exists(model["path"]), f"Invalid model path {model['path']}"
    assert isinstance(model["tp"], int), f"Invalid tp {model['tp']}"

    run_kwargs = {
        "path": model["path"],
        "tp": model["tp"],
        "engine": engine,
        "chip": chip,
        "image": image,
        "tokenizer": model["path"],
        "blocks": model.get("blocks", None),
        "max_model_len": model.get("max_model_len", 4096),
        "enable_prefix_caching": model.get("enable_prefix_caching", False),
        "quantization_method": model.get("quantization_method", None),
        "fim_template": model.get("fim_template", None)
    }

    return run_kwargs


def run_container(model, **kwargs):
    # Initialize configuration
    config = {
        "name": model,
        "image": f"{kwargs['image']}",
        "environment": {
            "MODEL": model,
            "TP": kwargs["tp"],
            "LOG": "1",
            "PREFIXCACHING": "True" if kwargs["enable_prefix_caching"] else "False"
        },
        "ports": {
            "8000/tcp": 8000,       # opanai api server port
            "20000/tcp": 20000,     # metrics server port
            "9000/tcp": 9000        # health check server port
        },
        "volumes": {
            f"{kwargs['path']}": { "bind": f"/app/model/{model}", "mode": "ro" }
        }
    }
    
    if kwargs["max_model_len"]:
        config["environment"]["MML"] = kwargs["max_model_len"]
    if kwargs["blocks"]:
        config["environment"]["BLOCKS"] = kwargs["blocks"]
    if kwargs["quantization_method"]:
        config["environment"]["QUANT"] = kwargs["quantization_method"]
    if kwargs["fim_template"]:
        config["environment"]["FIM_TEMPLATE"] = kwargs["fim_template"]
    
    # Automatically try to get available GPU for NVIDIA cards
    if kwargs["chip"] == "nvidia":
        if set_cuda_visible_devices(kwargs["tp"]) == False:
            sys.exit(1)
        
    # Set GPU
    if kwargs["chip"] == "enflame":
        config["environment"]["TOPS_VISIBLE_DEVICES"] = os.environ.get("TOPS_VISIBLE_DEVICES", None)
    else:
        config["environment"]["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", None)

    # engine related
    if kwargs["engine"] == "llm":
        config["environment"]["INFI_LOG_LEVEL"] = 3
    elif kwargs["engine"] == "sglang":
        config["environment"]["sglang_LOG_LEVEL"] = 3
    elif kwargs["engine"] == "vllm":
        pass
    else:
        logger.error(f"Invalid engine {kwargs['engine']}")
        sys.exit(1)
        
    # chip related
    if kwargs["chip"] == "nvidia":
        config["device_requests"] = [{
            "driver": "nvidia",
            "count": -1,
            "capabilities": [["gpu"]]
        }]
    elif kwargs["chip"] == "amd":
        config["devices"] = ["/dev/kfd", "/dev/dri"]
    elif kwargs["chip"] == "iluvatar":
        pass
    elif kwargs["chip"] == "enflame":
        pass
    elif kwargs["chip"] == "metax":
        config["devices"] = ["/dev/mxcd", "/dev/dri"]
        config["group_add"] = ["video"]
    elif kwargs["chip"] == "biren":
        pass
    elif kwargs["chip"] == "moorethreads":
        pass
    else:
        logger.error(f"Invalid chip {kwargs['chip']}")
        sys.exit(1)

    logger.info(f"Docker config = {config}")
    client = docker.from_env()
    try:
        container = client.containers.run(**config, detach=True)
        logger.info(f"Container ({container.name}) run successfully")
        logger.info(f"Container ID: {container.id}")
    except docker.errors.ImageNotFound:
        logger.error(f"Docker image ({kwargs['image']}) not found")
    except Exception as e:
        logger.error(f"Container ({kwargs['image']}) run failed with {e}");


def remove_container(model):
    try:
        client = docker.from_env()
        container = client.containers.get(model)  # get all containers
        logger.info(f"Container ({model}) found and remove ...")
        container.stop()
        container.remove()
        logger.info(f"Container ({model}) removed successfully")
    except docker.errors.NotFound:
        logger.error(f"Container ({model}) not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Container ({model}) remove failed with {e}");
        sys.exit(1)


def get_container_status(model):
    health_endpoint = "http://localhost:9000/health"
    # check container active
    active = False
    client = docker.from_env()
    for active_container in client.containers.list():
        if active_container.name == model:
            active = True

    # check service connected
    connected = False
    if active:
        try:
            response = requests.get(health_endpoint)
            response.raise_for_status()
            connected = True
        except requests.exceptions.RequestException as e:
            pass

    status = { "active": active, "connected": connected }
    return status


# Quick Start Test
Inference Image Testing (including sglang, vllm, lmdeploy engines)

## Option 1: vllm Inference Service Image Testing
### 1. Dependency Installation
```Bash
# Step.1 Recommended to use python3.10 environment
conda create --name llm_engine_test python=3.10
conda activate llm_engine_test
# Step.2 Install testing framework dependencies
cd llm_engine_test
pip install -r requirements.txt
```
### 2. Automated Testing
Taking NVIDIA-related vllm images as an example, the following commands can automatically pull corresponding images from harbor based on tags, and start corresponding model inference service container tests according to configuration files, then delete containers after testing. Test reports are saved in the llm_engine_test/reports/ directory.

#### Test All Models Under Configuration
```Bash
# Test sglang-nvidia image
python auto_run_test.py --models-yaml conf/vllm_sglang_nvidia_models.yaml --image-tag v0.1
# Test vllm-nvidia image
python auto_run_test.py --models-yaml conf/vllm_vllm_nvidia_models.yaml --image-tag v3.8.3
```
#### Test Single Model Under Configuration
```Bash
python auto_run_test.py --models-yaml conf/vllm_sglang_nvidia_models.yaml --image-tag v0.1 --model-id llama-3-8b-instruct
```
#### Adding Test Models, Hardware, Engines
Add/modify corresponding yaml configurations under llm_engine_test/conf/ directory
```YAML
engine: sglang  # Engine name: aligned with image name, supports sglang, vllm
device: nvidia  # Hardware name: aligned with image name
use_docker: True    # Default true for vllm image testing, otherwise use engine-specific SDK to start

parentdir: /share/datasets/public_models/   # Model root directory
models:
  - id: llama-2-7b-chat         # Requested model name
    tp: 1                       # TP count
    subdir: Llama-2-7b-chat-hf  # Model subdirectory
    blocks: 2000                # Serving startup blocks setting
    max_model_len: 4096         # Serving startup mml setting
  - id: qwen-1.5-14b-chat-fp8
    tp: 1
    subdir: /share/datasets/tmp_share/chenyonghua/qwen/Qwen1.5_14B_Chat_FP8
    path: /share/datasets/tmp_share/chenyonghua/qwen/Qwen1.5_14B_Chat_FP8        # Model path, if path is provided it will override parentdir and subdir
    blocks: 1800
    max_model_len: 25600
    quantization_method: fp8    # Quantization method, corresponding to quantized model
```
#### When Starting Docker Service Manually
If you want to start docker service manually for debugging, remember to map all corresponding ports, refer to the following:
```Bash
docker run --gpus all -tid \
           -p 8000:8000 -p 21002:21002 -p 20000:20000 -p 9000:9000 \
           -e BLOCKS=2049 -e MML=32768 \
           -e STOP_TOKEN_IDS=[151643,151644,151645] \
           -e MODEL=qwen2-7b-instruct -e TP=1 -e LOG=1 -e CUDA_VISIBLE_DEVICES=0 \
           -v /mnt/resource/public_models/Qwen_Qwen2-7B-Instruct:/app/model/qwen2-7b-instruct \
           llm-nvidia:v4.8
```
Execute corresponding case tests using pytest
```Bash
pytest tests/vllm/ tests/common/ -vvs \
    --model qwen2-7b-instruct \
    --model-path /mnt/resource/public_models/Qwen_Qwen2-7B-Instruct \
    --max-model-len 32768 \
    --engine vllm \
    --use-docker
```
(However, there will be no HTML report to view this way, you need to pay attention to terminal logs)

## Option 2: sglang-LLM SDK Testing
### 1. Self-developed Engine Environment Installation

### 2. Dependency Installation
```Bash
# Step.1 Activate pre-installed self-developed engine environment
conda activate sglang_env
# Step.2 Install testing framework dependencies
cd llm_engine_test
pip install -r requirements.txt
```

### 3. Execute Testing
#### Automated Testing
Automatically monitor GPU resources, start services, execute cases, and collect reports. Test reports are saved in the current directory report/
```Bash
# Execute all cases with default Serving startup parameters, by default loading conf/sglang_models.yaml
python auto_run_test.py
# Test only a specific model in conf/sglang_models.yaml
python auto_run_test.py --model-id llama-3-8b-instruct
# Execute other feature cases
python auto_run_test.py --models-yaml conf/xxx_models.yaml
# Start testing using docker image
docker pull sglang-nvidia:v0
python auto_run_test.py --models-yaml conf/vllm_sglang_nvidia_models.yaml --image-tag v0
# Test prefix caching, using docker as an example
python auto_run_test.py --models-yaml conf/vllm_sglang_prefix_caching_models.yaml --image-tag v0
```

#### Single Case Debugging
Service already started, test part of cases separately for debugging
```Bash
# Execute all cases under directory
pytest -vs tests/sglang/ --model xxx
# Execute specified test module
pytest -vs tests/sglang/test_chat_completions_api.py --model xxx
# Execute specified test class
pytest -vs tests/sglang/test_chat_completions_api.py::TestV1ChatCompletions --model xxx
# Execute specified test method
pytest -vs tests/sglang/test_chat_completions_api.py::TestV1ChatCompletions::test_chat_single_round --model xxx
# Execute specified priority cases
pytest -vs tests/sglang/test_chat_completions_api.py -m "P0 or P1" --model xxx
```


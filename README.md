# Quick Start Test
推理镜像测试（包含sglang、vllm、lmdeploy引擎）

## Option 1: vllm推理服务镜像测试
### 1. 依赖安装
```Bash
# Step.1 推荐使用python3.10环境
conda create --name llm_engine_test python=3.10
conda activate llm_engine_test
# Step.2 安装测试框架依赖
cd llm_engine_test
pip install -r requirements.txt
```
### 2. 自动化测试
以NVIDIA相关vllm镜像为例，以下命令可自动从harbor根据tag拉取对应镜像，并根据配置文件依次启动对应模型推理服务容器测试，测试后删除容器。测试报告保存在 llm_engine_test/reports/ 目录下

#### 测试配置下全部模型
```Bash
# 测试sglang-nvidia镜像
python auto_run_test.py --models-yaml conf/vllm_sglang_nvidia_models.yaml --image-tag v0.1
# 测试vllm-nvidia镜像
python auto_run_test.py --models-yaml conf/vllm_vllm_nvidia_models.yaml --image-tag v3.8.3
```
#### 测试配置下单个模型
```Bash
python auto_run_test.py --models-yaml conf/vllm_sglang_nvidia_models.yaml --image-tag v0.1 --model-id llama-3-8b-instruct
```
#### 新增测试模型、硬件、引擎
新增/修改 llm_engine_test/conf/ 目录下对应yaml配置即可
```YAML
engine: sglang  # 引擎名：与镜像名对齐，支持sglang、vllm
device: nvidia  # 硬件名：与镜像名对齐
use_docker: True    # 测试vllm镜像默认为true，否则使用引擎对应sdk启动

parentdir: /share/datasets/public_models/   # 模型根目录
models:
  - id: llama-2-7b-chat         # 请求的model name
    tp: 1                       # tp数   
    subdir: Llama-2-7b-chat-hf  # 模型子目录
    blocks: 2000                # Serving启动blocks设置
    max_model_len: 4096         # Serving启动mml设置
  - id: qwen-1.5-14b-chat-fp8
    tp: 1
    subdir: /share/datasets/tmp_share/chenyonghua/qwen/Qwen1.5_14B_Chat_FP8
    path: /share/datasets/tmp_share/chenyonghua/qwen/Qwen1.5_14B_Chat_FP8        # 模型路径，如果传了path会覆盖parentdir和subdir
    blocks: 1800
    max_model_len: 25600
    quantization_method: fp8    # 量化方法，与量化模型对应
```
#### 自己启动docker服务的情况
如果是想自己启动docker服务做一些调试，需要记得把对应端口都映射出来，参考下述
```Bash
docker run --gpus all -tid \
           -p 8000:8000 -p 21002:21002 -p 20000:20000 -p 9000:9000 \
           -e BLOCKS=2049 -e MML=32768 \
           -e STOP_TOKEN_IDS=[151643,151644,151645] \
           -e MODEL=qwen2-7b-instruct -e TP=1 -e LOG=1 -e CUDA_VISIBLE_DEVICES=0 \
           -v /mnt/resource/public_models/Qwen_Qwen2-7B-Instruct:/app/model/qwen2-7b-instruct \
           llm-nvidia:v4.8
```
使用pytest执行对应case测试
```Bash
pytest tests/vllm/ tests/common/ -vvs \
    --model qwen2-7b-instruct \
    --model-path /mnt/resource/public_models/Qwen_Qwen2-7B-Instruct \
    --max-model-len 32768 \
    --engine vllm \
    --use-docker 
```
(不过这样暂时就没有html报告可看了，需要自己关注终端日志)

## Option 2: sglang-LLM sdk测试
### 1. 自研引擎环境安装

### 2. 依赖安装
```Bash
# Step.1 激活前置安装的自研引擎环境
conda activate sglang_env
# Step.2 安装测试框架依赖
cd llm_engine_test
pip install -r requirements.txt
```

### 3. 执行测试
#### 自动化测试
自动监测显卡资源、启动服务、执行case、收集报告。测试报告保存在当前目录 report/ 下
```Bash
# 执行默认Serving启动参数的全部case，默认加载 conf/sglang_models.yaml
python auto_run_test.py
# 只测试 conf/sglang_models.yaml 中的某个模型
python auto_run_test.py --model-id llama-3-8b-instruct
# 执行其他feature的case
python auto_run_test.py --models-yaml conf/xxx_models.yaml
# 使用docker镜像启动测试
docker pull sglang-nvidia:v0
python auto_run_test.py --models-yaml conf/vllm_sglang_nvidia_models.yaml --image-tag v0
# 测试prefix caching，以docker为例
python auto_run_test.py --models-yaml conf/vllm_sglang_prefix_caching_models.yaml --image-tag v0
```

#### 单case调试
服务已启动，单独测试部分case，调试用
```Bash
# 执行目录下全部case
pytest -vs tests/sglang/ --model xxx
# 执行指定测试模块
pytest -vs tests/sglang/test_chat_completions_api.py --model xxx
# 执行指定测试类     
pytest -vs tests/sglang/test_chat_completions_api.py::TestV1ChatCompletions --model xxx
# 执行指定测试方法    
pytest -vs tests/sglang/test_chat_completions_api.py::TestV1ChatCompletions::test_chat_single_round --model xxx
# 执行指定优先级case    
pytest -vs tests/sglang/test_chat_completions_api.py -m "P0 or P1" --model xxx
```


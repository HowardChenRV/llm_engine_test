## 下载cuda镜像
docker pull liwen/nvidia/cuda@sha256:20f6332cd4597673d3324f6740f33e02b0671c48bc80b058d994d38ffeb7716a

docker tag d0117ee15b5f nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

## 执行编译
docker build -t sglang_builder:cuda11.8.0-cudnn8-devel-ubuntu22.04 -f Dockerfile_sglang_builder .


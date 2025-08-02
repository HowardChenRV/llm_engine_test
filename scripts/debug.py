from ..utils.llm_server_api import llmServerApi

from openai import OpenAI
import json


def test_llm_api():
    llmServerApi.ping_generate("127.0.0.1", "9001")
    llmServerApi.ping_health("127.0.0.1", "8000")
    llmServerApi.ping_metrics("127.0.0.1", "8000")
    llmServerApi.ping_generate_stream("127.0.0.1", "9001")



def test_openai_api():
    model = "/share/datasets/public_models/Meta-Llama-3-8B-Instruct"
    file_path = "/share/datasets/tmp_share/chenyonghua/datasets/long_bench/long_bench-qwen2-8k_32k.json"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        datasets = json.load(file)
    
    openai_cli = OpenAI(
        api_key="Bearer test_api_key",
        base_url=f"http://127.0.0.1:9001/v1"
    )
    
    # for dataset in datasets:
    #     if dataset["prompt_len"] > 8000 and dataset["prompt_len"] < 10000:
    #         question = dataset
    #         break

    question = {
        "prompt": "hello",
        "output_len": 128
    }
    
    print(question)
    
    request_content = {
        "model": model,
        "max_tokens": question["output_len"],
        "messages": [
            { "role": "user", "content": question["prompt"] }
        ],
        "stream": True,
        "stream_options": { "include_usage": True }
    }
            
    stream = openai_cli.chat.completions.create(
        **request_content
    )

    for chunk in stream:
        print(chunk)
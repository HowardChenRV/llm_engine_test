import json

import pytest
import requests
from transformers import AutoTokenizer

from ...utils.logger import logger


API_SERVER_URL = "http://localhost:8000"


class TestllmGenerate:
    url = f"{API_SERVER_URL}/generate"
    headers = {"Content-Type": "application/json"}
    tokenizer = AutoTokenizer.from_pretrained("/share/datasets/public_models/Qwen_Qwen1.5-72B-Chat/", trust_remote_code=True)
    messages = [
        {'role': 'user', 'content': 'what is the largest animal in the world?'},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    answer = "blue whale"

    @pytest.mark.P0
    def test_chat_single_round(self):
        max_tokens = 50
        message = [
            {'role': 'user', 'content': '你是谁？'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        data_json = response.json()
        text_list = data_json.get("text", [])
        out_len = data_json.get("out_len", None)
        assert out_len == max_tokens and len(text_list) == 1
        text = text_list[0]
        assert "assistant" in text.lower(), f"test_chat_single_round fail, text = {text}"

    @pytest.mark.P0
    def test_chat_multi_round(self):
        max_tokens = 50
        message = [
            {'role': 'user', 'content': '你是谁？'},
            {'role': 'assistant', 'content': '我是大模型回答助手'},
            {'role': 'user', 'content': '你能做什么？'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        data_json = response.json()
        text_list = data_json.get("text", [])
        out_len = data_json.get("out_len", None)
        assert out_len == max_tokens and len(text_list) == 1
        text = text_list[0]
        assert "assistant" in text.lower(), f"test_chat_multi_round fail, text = {text}"

    @pytest.mark.P0
    def test_chat_system_prompt(self):
        max_tokens = 50
        message = [
            {'role': 'system', 'content': '请以嘲讽的语气回答'},
            {'role': 'user', 'content': '你是谁？'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": True,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        data_json = response.json()
        text_list = data_json.get("text", [])
        out_len = data_json.get("out_len", None)
        assert out_len == max_tokens and len(text_list) == 1
        text = text_list[0]
        assert "assistant" in text.lower(), f"test_chat_system_prompt fail, text = {text}"

    @pytest.mark.P0
    def test_chat_with_stream(self):
        max_tokens = 50
        messages = [
            {'role': 'user', 'content': '你是谁？'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "stream": True,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        text = ""
        iter_count = 1
        for line in response.iter_lines(decode_unicode=True):
            if line:
                logger.debug(f"Stream Iterate: {line}")
                data_json = json.loads(line)
                text_list = data_json.get("text", [])
                out_len = data_json.get("out_len", None)
                assert out_len == iter_count
                assert len(text_list) == 1
                text = text_list[0]
                assert text is not None and text != ""
                iter_count += 1

        assert "assistant" in text.lower(), f"test_chat_with_stream fail, text = {text}"

    @pytest.mark.P0
    def test_chat_given_stop(self):
        max_tokens = 50
        message = [
            {'role': 'user', 'content': 'what is the largest animal in the world?'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": False,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        data_json = response.json()
        text_list = data_json.get("text", [])
        out_len = data_json.get("out_len", None)
        assert out_len == max_tokens and len(text_list) == 1
        text_max_len_50 = text_list[0]

        # given stop
        stop_words = ["blue", "qwer"]
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "ignore_eos": False,
            "stop": stop_words,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        data_json = response.json()
        text_list = data_json.get("text", [])
        out_len = data_json.get("out_len", None)
        assert out_len < max_tokens and len(text_list) == 1
        text_stop = text_list[0]

        assert text_max_len_50.replace(text_stop, "").split(" ")[0] in stop_words

    @pytest.mark.P0
    def test_chat_temperature_0(self):
        max_tokens = 50
        message = [
            {'role': 'user', 'content': 'what is the largest animal in the world?'},
        ]
        prompt = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt,
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 10,
            "max_tokens": max_tokens,
            "ignore_eos": False,
        }
        logger.info(f"Try call url body: {data}")
        response_1 = requests.post(self.url, headers=self.headers, json=data)
        response_2 = requests.post(self.url, headers=self.headers, json=data)
        assert response_1.status_code == 200 and response_2.status_code == 200
        logger.info(response_1.text)
        logger.info(response_2.text)

        text_1 = response_1.json().get("text", [])[0]
        text_2 = response_2.json().get("text", [])[0]

        assert self.answer in text_1.lower() and self.answer in text_2.lower()
        assert text_1 == text_2, f"test_chat_temperature_0 fail, text1 is {text_1} and text2 is {text_2}"

    @pytest.mark.P1
    def test_ignore_eos(self):
        messages_s = [
            {'role': 'user', 'content': 'What is the square root of 81?'},
        ]
        prompt_s = self.tokenizer.apply_chat_template(
            messages_s,
            tokenize=False,
            add_generation_prompt=True
        )
        data = {
            "prompt": prompt_s,
            "temperature": 0.0,
            "max_tokens": 100,
            "ignore_eos": True,
        }
        logger.info(f"Try call url body: {data}")
        response = requests.post(self.url, headers=self.headers, json=data)
        assert response.status_code == 200
        logger.info(response.text)

        data_json = response.json()
        text_list = data_json.get("text", [])
        out_len = data_json.get("out_len", None)
        assert out_len == 100 and len(text_list) == 1

    @pytest.mark.P1
    @pytest.mark.parametrize("temper", [2, 0.8])
    def test_sample_temperature(self, temper):
        data = {
            "prompt": self.prompt,
            "temperature": temper,
            "max_tokens": 50,
        }
        logger.info(f"Try call url body: {data}")
        response_1 = requests.post(self.url, headers=self.headers, json=data)
        response_2 = requests.post(self.url, headers=self.headers, json=data)
        assert response_1.status_code == 200 and response_2.status_code == 200
        logger.info(response_1.text)
        logger.info(response_2.text)

        text_1 = response_1.json().get("text", [])[0]
        text_2 = response_2.json().get("text", [])[0]

        assert text_1 != text_2, f"校验随机采样，也有可能相等，可酌情跳过该报错"
        if temper < 1:
            assert self.answer in text_1.lower() and self.answer in text_2.lower()

    @pytest.mark.P1
    @pytest.mark.parametrize("top_p", [1])
    def test_sample_top_p(self, top_p):
        data = {
            "prompt": self.prompt,
            "max_tokens": 50,
            "top_p": top_p,
        }
        logger.info(f"Try call url body: {data}")
        response_1 = requests.post(self.url, headers=self.headers, json=data)
        response_2 = requests.post(self.url, headers=self.headers, json=data)
        assert response_1.status_code == 200 and response_2.status_code == 200
        logger.info(response_1.text)
        logger.info(response_2.text)

        text_1 = response_1.json().get("text", [])[0]
        text_2 = response_2.json().get("text", [])[0]

        assert self.answer in text_1.lower() and self.answer in text_2.lower()
        assert text_1 != text_2, f"校验随机采样，也有可能相等，可酌情跳过该报错"

    @pytest.mark.P1
    @pytest.mark.parametrize("top_k", [10])
    def test_sample_top_k(self, top_k):
        data = {
            "prompt": self.prompt,
            "max_tokens": 50,
            "top_k": top_k,
        }
        logger.info(f"Try call url body: {data}")
        response_1 = requests.post(self.url, headers=self.headers, json=data)
        response_2 = requests.post(self.url, headers=self.headers, json=data)
        assert response_1.status_code == 200 and response_2.status_code == 200
        logger.info(response_1.text)
        logger.info(response_2.text)

        text_1 = response_1.json().get("text", [])[0]
        text_2 = response_2.json().get("text", [])[0]

        assert self.answer in text_1.lower() and self.answer in text_2.lower()
        assert text_1 != text_2, f"校验随机采样，也有可能相等，可酌情跳过该报错"

    @pytest.mark.P1
    def test_sample_union(self):
        data = {
            "prompt": self.prompt,
            "max_tokens": 50,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 10,
        }
        logger.info(f"Try call url body: {data}")
        response_1 = requests.post(self.url, headers=self.headers, json=data)
        response_2 = requests.post(self.url, headers=self.headers, json=data)
        assert response_1.status_code == 200 and response_2.status_code == 200
        logger.info(response_1.text)
        logger.info(response_2.text)

        text_1 = response_1.json().get("text", [])[0]
        text_2 = response_2.json().get("text", [])[0]

        assert self.answer in text_1.lower() and self.answer in text_2.lower()
        assert text_1 != text_2, f"校验随机采样，也有可能相等，可酌情跳过该报错"

    @pytest.mark.P1
    @pytest.mark.parametrize("args", [[2, 0.001, 10], [2, 1, 1]])   # temperature, top_p, top_k
    def test_greedy_search(self, args):
        temperature, top_p, top_k = args[0], args[1], args[2]
        data = {
            "prompt": self.prompt,
            "max_tokens": 50,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }
        logger.info(f"Try call url body: {data}")
        response_1 = requests.post(self.url, headers=self.headers, json=data)
        response_2 = requests.post(self.url, headers=self.headers, json=data)
        assert response_1.status_code == 200 and response_2.status_code == 200
        logger.info(response_1.text)
        logger.info(response_2.text)

        text_1 = response_1.json().get("text", [])[0]
        text_2 = response_2.json().get("text", [])[0]

        assert self.answer in text_1.lower() and self.answer in text_2.lower()
        assert text_1 == text_2

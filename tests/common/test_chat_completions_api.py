import sys
import traceback
import pytest
from ...utils.correctness_tool import CorrectnessTool
from ...utils.openai_tool import OpenAIResponseValidator
from ...utils.logger import logger


# Ordered by official OpenAI API documentation
# https://platform.openai.com/docs/api-reference/chat/create
class TestV1ChatCompletions:
    
    @pytest.mark.P0
    @pytest.mark.description("Dialog reasoning test - Non-stream mode: Single round dialogue")
    def test_chat_single_round(self, client, model):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ]
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P0
    @pytest.mark.description("Dialog reasoning test - Non-stream mode: Multi-round dialogue")
    def test_chat_multi_round(self, client, model):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": "Hi!" },
                { "role": "assistant", "content": "Hi!" },
                { "role": "user", "content": question["prompt"] }
            ]
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P0
    @pytest.mark.description("Dialog reasoning test - Non-stream mode: system_prompt")
    def test_chat_system_prompt(self, client, model):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "system", "content": "Please act as a university professor." },
                { "role": "user", "content": question["prompt"] }
            ]
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P0
    @pytest.mark.description("Dialog reasoning test - Non-stream mode: Long text")
    def test_chat_long_context(self, client, model, model_config, max_model_len):
        prompt_len = 4096
        if model_config:
            if max_model_len < model_config.max_position_embeddings:
                prompt_len = max_model_len
            else:
                prompt_len = model_config.max_position_embeddings
                
        request_content = {
            "model": model,
            "max_tokens": 128,
            "messages": [
                { "role": "user", "content": "Hi" * (prompt_len - 512) }
            ]
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            assert len(open_ai_response_validator.content[0])>0, f"Long text test failed, no valid content returned: {chat_completion}"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P1
    @pytest.mark.description("Dialog reasoning test - Non-stream mode: n > 1")
    def test_chat_with_n(self, client, model, engine, use_docker):
        if not use_docker and engine in ["sglang", "vllm", "lmdeploy"]:
            pytest.skip(f"{engine} do not support n > 1")
            
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "n": 2
        }

        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
            CorrectnessTool.check_answer(open_ai_response_validator.content[1])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    # Core functionality - Dialog reasoning test - Stream
    @pytest.mark.P0
    @pytest.mark.description("Dialog reasoning test - Stream mode: include_usage == True or False")
    @pytest.mark.parametrize("include_usage", [False, True])
    def test_chat_with_stream(self, client, model, include_usage):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "stream": True,
        }
        if include_usage:
            request_content["stream_options"] = { "include_usage": True }
                
        try:
            stream = client.chat.completions.create(
                **request_content
            )
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_stream_response_structure(response=stream)
            logger.debug(open_ai_response_validator)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P0
    @pytest.mark.description("Dialog reasoning test - Stream mode: Long text")
    def test_chat_with_stream_long_context(self, client, model, model_config, max_model_len):
        prompt_len = 128
        if model_config:
            if max_model_len < model_config.max_position_embeddings:
                prompt_len = max_model_len
            else:
                prompt_len = model_config.max_position_embeddings
                
        request_content = {
            "model": model,
            "max_tokens": 10,
            "messages": [
                { "role": "user", "content": "Hi" * (prompt_len - 512) }
            ],
            "stream": True,
            "stream_options": { "include_usage": True }
        }

                
        try:
            stream = client.chat.completions.create(
                **request_content
            )
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_stream_response_structure(response=stream)
            logger.debug(open_ai_response_validator)
            assert len(open_ai_response_validator.content[0])>0, f"Long text test failed, no valid content returned: {open_ai_response_validator}"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P1
    @pytest.mark.description("Dialog reasoning test - Stream mode: n > 1")
    def test_chat_with_stream_and_n(self, client, model, engine, use_docker):
        if not use_docker and engine in ["sglang", "vllm", "lmdeploy"]:
            pytest.skip(f"{engine} do not support n > 1")
            
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "stream": True,
            "stream_options": { "include_usage": True },
            "n": 2
        }
       
        try:
            stream = client.chat.completions.create(
                **request_content
            )
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_stream_response_structure(response=stream)
            logger.debug(open_ai_response_validator)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
            CorrectnessTool.check_answer(open_ai_response_validator.content[1])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
    
    
    @pytest.mark.P0
    @pytest.mark.description("Termination parameter test - stop")
    @pytest.mark.parametrize("stop_param", [["5"], ["10", "9", "8", "7", "6", "5"]])
    def test_chat_with_stop(self, client, model, stop_param):
        request_content = {
            "model": model,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 20. Note: Just respond with the numbers only." } 
            ],
            "temperature": 0,
            "stop": stop_param
        } 
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            # Verify termination content
            assert chat_completion.choices[0].finish_reason == "stop"
            for num in ["1", "2", "3", "4"]:
                assert num in open_ai_response_validator.content[0], \
                    f"chat with stop error, {num} not in : '{open_ai_response_validator.content[0]}'"
            for num in ["5", "6", "7", "8", "9", "10"]:
                assert num not in open_ai_response_validator.content[0], \
                    f"chat with stop error, found {num} in : {open_ai_response_validator.content[0]}"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            
            
    @pytest.mark.skip(reason="vllm and most of engines do not limit stop num.")
    @pytest.mark.P2
    @pytest.mark.description("Termination parameter test - stop: Boundary value stop length > 4, expected error code interception")
    @pytest.mark.parametrize("stop_param", [["8", "7", "6", "5", "1"]]) # stop长度预期<=4
    def test_chat_with_stop_parameter_error(self, client, model, stop_param, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")
            
        request_content = {
            "model": model,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 20. Note: Just respond with the numbers only." } 
            ],
            "temperature": 0,
            "stop": stop_param
        }
        
        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40302)
        finally:
            assert catched


    @pytest.mark.P0
    @pytest.mark.description("Termination parameter test - max_tokens: max_tokens == 10")
    def test_chat_with_max_tokens(self, client, model):
        request_content = {
            "model": model,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 10. Note: Just respond with the numbers only." } 
            ],
            "temperature": 0,
            "max_tokens": 10
        } 
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            # Verify termination length
            assert chat_completion.choices[0].finish_reason == "length"
            assert open_ai_response_validator.usage[0].completion_tokens == 10, f"chat with max tokens error, {open_ai_response_validator.content[0]}"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P1
    @pytest.mark.description("Termination parameter test - stop combined with max_tokens: Expected to stop early")
    def test_chat_with_stop_and_max_tokens(self, client, model):
        request_content = {
            "model": model,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 1000. Note: Just respond with the numbers only." } 
            ],
            "temperature": 0,
            "stop": ["2"],
            "max_tokens": 4096
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            # Verify termination content
            assert chat_completion.choices[0].finish_reason == "stop"
            assert open_ai_response_validator.usage[0].completion_tokens < 64, f"chat with stop and max tokens error, {open_ai_response_validator.content[0]}"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            

    @pytest.mark.P1
    @pytest.mark.description("Sampling strategy test - Temperature sampling: temperature normal values, special values, extreme max/min values")
    @pytest.mark.parametrize("temperature", [0.0001, 0.00001, 0.000001, 0.8, 2, 1.9999])
    def test_chat_random_response_with_temperature(self, client, model, temperature):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "temperature": temperature
        }
        
        try:
            validator_list = []
            # 请求两次
            for _ in range(2):
                chat_completion = client.chat.completions.create(**request_content)
                open_ai_response_validator = OpenAIResponseValidator(**request_content)
                open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
                if temperature < 1: 
                    CorrectnessTool.check_answer(open_ai_response_validator.content[0])
                validator_list.append(open_ai_response_validator)
            # Verify random sampling, may also be equal, can optionally skip this error
            if temperature > 0.1:
                assert validator_list[0].content[0] != validator_list[1].content[0], \
                    f"Verifying random sampling, they may also be equal, you can optionally skip this error"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            
            
    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P1
    @pytest.mark.description("Sampling strategy test - Temperature sampling: temperature boundary values, expected error code interception")
    @pytest.mark.parametrize("temperature", [-1, 2.1])
    def test_chat_with_temperature_parameter_error(self, client, model, temperature, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")
            
        request_content = {
            "model": model,
            "max_tokens": 64,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 1000." }
            ],
            "temperature": temperature
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40302)
        finally:
            assert catched


    @pytest.mark.P1
    @pytest.mark.description("Sampling strategy test - Top_p sampling: top_p normal values, extreme max/min values")
    @pytest.mark.parametrize("top_p", [0.001, 0.6, 1, 0.999])
    def test_chat_random_response_with_top_p(self, client, model, top_p):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "top_p": top_p
        }
        
        try:
            validator_list = []
            # 请求两次
            for _ in range(2):
                chat_completion = client.chat.completions.create(**request_content)
                open_ai_response_validator = OpenAIResponseValidator(**request_content)
                open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
                CorrectnessTool.check_answer(open_ai_response_validator.content[0])
                validator_list.append(open_ai_response_validator)
            # Verify random sampling, may also be equal, can optionally skip this error
            if top_p > 0.1:
                assert validator_list[0].content[0] != validator_list[1].content[0], \
                    f"校验随机采样，也有可能相等，可酌情跳过该报错"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            

    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P1
    @pytest.mark.description("采样策略测试-Top_p采样: top_p边界值, 预期错误码拦截")
    @pytest.mark.parametrize("top_p", [-1, 1.1])
    def test_chat_with_top_p_parameter_error(self, client, model, top_p, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")
            
        request_content = {
            "model": model,
            "max_tokens": 64,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 1000." }
            ],
            "top_p": top_p
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40302)
        finally:
            assert catched


    @pytest.mark.P1
    @pytest.mark.description("采样策略测试-联合采样: temperature + top_p")
    def test_chat_random_response_with_different_parameter(self, client, model):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "temperature": 0.8,
            "top_p": 0.9
        }
        
        try:
            validator_list = []
            # 请求两次
            for _ in range(2):
                chat_completion = client.chat.completions.create(**request_content)
                open_ai_response_validator = OpenAIResponseValidator(**request_content)
                open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
                CorrectnessTool.check_answer(open_ai_response_validator.content[0])
                validator_list.append(open_ai_response_validator)
            # 校验随机采样，也有可能相等，可酌情跳过该报错
            assert validator_list[0].content[0] != validator_list[1].content[0], \
                f"Verifying random sampling, they may also be equal, you can optionally skip this error"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            

    @pytest.mark.skip(reason="No longer supported")
    @pytest.mark.P1
    @pytest.mark.description("采样策略测试-Top_k采样: top_k = [10, 50, 100, 1000]")
    @pytest.mark.parametrize("top_k", [10, 50, 100, 1000])
    def test_chat_random_response_with_top_k(self, client, model, top_k, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports top_k cases.")
        
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "top_k": top_k
        }
        
        try:
            validator_list = []
            # 请求两次
            for _ in range(2):
                chat_completion = client.chat.completions.create(**request_content)
                open_ai_response_validator = OpenAIResponseValidator(**request_content)
                open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
                CorrectnessTool.check_answer(open_ai_response_validator.content[0])
                validator_list.append(open_ai_response_validator)
            # 校验随机采样，也有可能相等，可酌情跳过该报错
            assert validator_list[0].content[0] != validator_list[1].content[0], \
                f"Verifying random sampling, they may also be equal, you can optionally skip this error"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.skip(reason="No longer supported")
    @pytest.mark.P2
    @pytest.mark.description("采样策略测试-Top_k采样: top_k特殊值, 预期错误码拦截")
    @pytest.mark.parametrize("top_k", [0])
    def test_chat_with_top_k_parameter_error(self, client, model, top_k, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")
            
        request_content = {
            "model": model,
            "max_tokens": 64,
            "messages": [
                { "role": "user", "content": "Please count from 1 to 1000." }
            ],
            "top_k": top_k
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=400)
        finally:
            assert catched
            
    
    @pytest.mark.P1
    @pytest.mark.description("采样策略测试-Greedy Search: temperature + top_p")
    @pytest.mark.parametrize("temperature, top_p", [
        (0, 1),
        (2, 0.01),
        (0.000009, 1),  # vllm/llm隐含逻辑：小于_SAMPLING_EPS=0.00001，转为贪婪采样逻辑，内容正确
        ])
    def test_chat_with_greedy_search(self, client, model, temperature, top_p):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "temperature": temperature,
            "top_p": top_p
        }

        try:
            validator_list = []
            # 请求两次
            for _ in range(2):
                chat_completion = client.chat.completions.create(**request_content)
                open_ai_response_validator = OpenAIResponseValidator(**request_content)
                open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
                CorrectnessTool.check_answer(open_ai_response_validator.content[0])
                validator_list.append(open_ai_response_validator)
            # 校验随机采样，也有可能相等，可酌情跳过该报错
            assert validator_list[0].content[0] == validator_list[1].content[0], f"Greedy Search validation failed"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P1
    @pytest.mark.description("多样性参数测试-presence_penalty: 正常值、极大/极小值")
    @pytest.mark.parametrize("presence_penalty", [-2, -1.999, -1, 0, 1, 1.999, 2])
    def test_chat_with_presence_penalty(self, client, model, presence_penalty):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "presence_penalty": presence_penalty
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P1
    @pytest.mark.description("多样性参数测试-frequency_penalty: 正常值、极大/极小值")
    @pytest.mark.parametrize("frequency_penalty", [-2, -1.999, -1, 0, 1, 1.999, 2])
    def test_chat_with_frequency_penalty(self, client, model, frequency_penalty):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "frequency_penalty": frequency_penalty
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            
            
    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P2
    @pytest.mark.description("多样性参数测试-presence_penalty: 边界值, 预期错误码拦截")
    @pytest.mark.parametrize("presence_penalty", [-2.1, 2.1])
    def test_chat_with_presence_penalty_parameter_error(self, client, model, presence_penalty, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")
            
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "presence_penalty": presence_penalty
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40302)
        finally:
            assert catched


    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P2
    @pytest.mark.description("多样性参数测试-frequency_penalty: 边界值, 预期错误码拦截")
    @pytest.mark.parametrize("frequency_penalty", [-2.1, 2.1])
    def test_chat_with_frequency_penalty_parameter_error(self, client, model, frequency_penalty, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")

        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "frequency_penalty": frequency_penalty
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40302)
        finally:
            assert catched


    @pytest.mark.P1
    @pytest.mark.description("logprobs测试-logprobs: logprobs = True and top_logprobs = 2")
    def test_chat_with_logprobs(self, client, model, use_docker):
        if use_docker:
            pytest.skip(f"vllm image not supports logprobs, checkout at: https://docs.llm-ai.com/gen-studio/api/vllm.html#/schemas/ChatCompletionRequest")
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "messages": [
                { "role": "user", "content": question["prompt"] }
            ],
            "logprobs": True,
            "top_logprobs": 2
        }
        
        try:
            chat_completion = client.chat.completions.create(
                **request_content
            )
            logger.debug(chat_completion)
            open_ai_response_validator = OpenAIResponseValidator(**request_content)
            open_ai_response_validator._validate_non_stream_response_structure(response=chat_completion)
            CorrectnessTool.check_answer(open_ai_response_validator.content[0])            
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P1
    @pytest.mark.description("异常容错测试-必填参数未传: model、messages, 预期错误码拦截")
    @pytest.mark.parametrize("data", [
        {"messages": [ { "role": "user", "content": "Hi!" } ]},
        {"model": "model_id"},
    ])
    def test_chat_required_parameter_not_provided(self, client, data):
        request_content = data

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            assert isinstance(e, TypeError)
            logger.debug(e)
            catched = True
        finally:
            assert catched


    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P1
    @pytest.mark.description("异常容错测试-错误参数值测试: 错误model, 预期错误码拦截")
    def test_chat_with_invalid_model(self, client, model, use_docker):
        request_content = {
            "model": model + "x",
            "messages": [ { "role": "user", "content": "Hi!" } ]
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40301)
        finally:
            assert catched


    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P2
    @pytest.mark.description("异常容错测试-参数数据数据类型异常: stream、temperature、top_p、max_tokens、stop、presence_penalty、frequency_penalty, 预期错误码拦截")
    @pytest.mark.parametrize("aditional_data", [
        { "stream": "string" },
        { "temperature": "string" },
        { "top_p": "string" },
        # { "n": 1.1 },
        { "max_tokens": 1.1 },
        { "stop": [1] },
        { "presence_penalty": "string" },
        { "frequency_penalty": "string" },
        { "messages": [ { "content": "Hi!"} ] },
    ])
    def test_chat_with_invalid_data_type(self, client, model, aditional_data, use_docker):
        if not use_docker:
            pytest.skip(f"Only the vllm image supports such strict OpenAI protocol boundary cases.")
            
        request_content = {
            "model": model,
            "messages": [ { "role": "user", "content": "Hi!" } ]
        }
        request_content.update(aditional_data)

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40001)
        finally:
            assert catched


    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P2
    @pytest.mark.description("异常容错测试-特殊值: n、messages、, 预期错误码拦截")
    @pytest.mark.parametrize("aditional_data", [
        { "n": 0 },
        { "n": -1 },
        { "messages": [] },
    ])
    def test_chat_with_invalid_parameter(self, client, model, aditional_data, use_docker):
            
        request_content = {
            "model": model,
            "messages": [ { "role": "user", "content": "Hi!" } ]
        }
        request_content.update(aditional_data)

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40302)
        finally:
            assert catched


    # @pytest.mark.skip(reason="not support")
    @pytest.mark.P1
    @pytest.mark.description("异常容错测试-特殊值: prompt内容超过启动MML限制最大长度, 预期错误码拦截")
    def test_chat_with_mml_overflow(self, client, model, use_docker, max_model_len):
        prompt_len = 4096
        
        if prompt_len < max_model_len:
            prompt_len = max_model_len
            
        request_content = {
            "model": model,
            "max_tokens": 10,
            "messages": [
                { "role": "user", "content":  "Hi" * (prompt_len + 1)}
            ]
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.info(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40303)
        finally:
            assert catched
            

    @pytest.mark.P1
    @pytest.mark.description("异常容错测试-特殊值: prompt内容超过模型配置限制最大长度, 预期错误码拦截")
    def test_chat_with_context_overflow(self, client, model, use_docker, model_config):
        prompt_len = 4096
        if model_config:
            prompt_len = model_config.max_position_embeddings
            
        request_content = {
            "model": model,
            "max_tokens": 10,
            "messages": [
                { "role": "user", "content":  "Hi" * (prompt_len + 1)}
            ]
        }

        try:
            catched = False
            response = client.chat.completions.create(
                **request_content
            )
            logger.debug(response)
        except Exception as e:
            catched = True
            logger.debug(e)
            validator = OpenAIResponseValidator(**request_content)
            validator._validate_api_error(e=e, use_docker=use_docker, error_code=40303)
        finally:
            assert catched
import sys
import traceback
import pytest
from ...utils.correctness_tool import CorrectnessTool
from ...utils.openai_tool import OpenAIResponseValidator
from ...utils.logger import logger


# Ordered by official OpenAI API documentation
# https://platform.openai.com/docs/api-reference/completions/create
class TestV1Completions:
    
    @pytest.mark.P1
    @pytest.mark.description("Completions接口对话推理测试: 非Stream模式")
    def test_chat_single_round(self, client, model):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "prompt": question["prompt"],
            "temperature": 0
        }
        
        try:
            completion = client.completions.create(
                **request_content
            )
            logger.debug(completion)
            # 必须参数校验
            for attr in ["id", "object", "created", "model", "choices", "usage"]:
                assert hasattr(completion, attr), f'completion返回参数校验失败'
            assert completion.model == model
            # 校验choices
            assert len(completion.choices) > 0
            for choice in completion.choices:
                for attr in ["text", "index", "finish_reason"]:
                    assert hasattr(choice, attr), f'choice返回参数校验失败'
                assert choice.finish_reason in ["stop", "length"]
                CorrectnessTool.check_answer(choice.text)
            # 校验usage
            for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                assert hasattr(completion.usage, attr), f'usage返回参数校验失败'
            assert completion.usage.prompt_tokens>0 and completion.usage.completion_tokens>0
            assert completion.usage.completion_tokens <= question["max_tokens"]
            assert completion.usage.total_tokens == completion.usage.prompt_tokens + completion.usage.completion_tokens
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()


    @pytest.mark.P2
    @pytest.mark.description("Completions接口对话推理测试: Stream模式")
    def test_chat_with_stream(self, client, model):
        question = CorrectnessTool.get_question()
        request_content = {
            "model": model,
            "max_tokens": question["max_tokens"],
            "prompt": question["prompt"],
            "temperature": 0,
            "stream": True,
            "stream_options": { "include_usage": True }
        }
        
        try:
            stream = client.completions.create(
                **request_content
            )
            content = ""
            finish_reason = None
            for completion in stream:
                # print(completion)
                # 必须参数校验
                for attr in ["id", "object", "created", "model", "choices", "usage"]:
                    assert hasattr(completion, attr), f'completion返回参数校验失败'
                assert completion.model == model
                # 校验choices
                if finish_reason == None:
                    assert len(completion.choices) > 0
                for choice in completion.choices:
                    for attr in ["text", "index", "finish_reason"]:
                        assert hasattr(choice, attr), f'choice返回参数校验失败'
                    content += choice.text
                    finish_reason = choice.finish_reason
                # 校验usage
                for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                    assert hasattr(completion.usage, attr), f'usage返回参数校验失败'
                assert completion.usage.prompt_tokens>0 and completion.usage.completion_tokens>0
                assert completion.usage.completion_tokens <= question["max_tokens"]
                assert completion.usage.total_tokens == completion.usage.prompt_tokens + completion.usage.completion_tokens
            assert finish_reason in ["stop", "length"]
            CorrectnessTool.check_answer(content)
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()
            
    @pytest.mark.P2
    @pytest.mark.description("终止参数测试-stop 非Stream模式")
    @pytest.mark.parametrize("stop_param", [["5"], ["10", "9", "8", "7", "6", "5"]])
    def test_chat_with_stop(self, client, model, stop_param):
        request_content = {
            "model": model,
            "prompt": "Please count from 1 to 20. Note: Just respond with the numbers only.",
            "temperature": 0,
            "stop": stop_param
        } 
        
        try:
            completion = client.completions.create(
                **request_content
            )
            logger.debug(completion)
            # 校验终止内容
            assert completion.choices[0].finish_reason == "stop"
            for num in ["1", "2", "3", "4"]:
                assert num in completion.choices[0].text, \
                    f"chat with stop error, {num} not in : '{completion.choices[0].text}'"
            for num in ["5", "6", "7", "8", "9", "10"]:
                assert num not in completion.choices[0].text, \
                    f"chat with stop error, found {num} in : {completion.choices[0].text}"
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()

    @pytest.mark.P2
    @pytest.mark.description("FIM参数测试-suffix 非Stream模式")
    def test_chat_with_suffix(self, client, model, fim_template):
        if not fim_template:
            pytest.skip("FIM is not supported for this model.")
        suffix = "return fib(a-1) + fib(a-2)"
        request_content = {
            "model": model,
            "prompt": "def fib(a):",
            "temperature": 0,
            "max_tokens": 128,
            "suffix": suffix
        }
        try:
            completion = client.completions.create(
                **request_content
            )
            logger.debug(completion)
            # 校验终止内容
            assert len(completion.choices) > 0
            for choice in completion.choices:
                for attr in ["text", "index", "finish_reason"]:
                    assert hasattr(choice, attr), f'choice返回参数校验失败'
                assert choice.finish_reason == "stop"
                assert choice.text.endswith(suffix)
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()

    @pytest.mark.P2
    @pytest.mark.description("logprobs参数测试: 非Stream模式")
    def test_chat_with_logprobs(self, client, model):
        request_content = {
            "model": model,
            "prompt": "what is the largest animal in the world?",
            "temperature": 0,
            "max_tokens": 10,
            "logprobs": 4
        }
        try:
            completion = client.completions.create(
                **request_content
            )
            logger.debug(completion)
            # 校验logprobs内容
            assert len(completion.choices) > 0
            for choice in completion.choices:
                for attr in ["text", "index", "finish_reason", "logprobs"]:
                    assert hasattr(choice, attr), f'choice返回参数校验失败'
                assert len(choice.logprobs.top_logprobs[0]) == 4
                assert "".join(choice.logprobs.tokens) == choice.text
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()

    @pytest.mark.P2
    @pytest.mark.description("logprobs参数测试: Stream模式")
    def test_chat_with_logprobs(self, client, model):
        request_content = {
            "model": model,
            "prompt": "what is the largest animal in the world?",
            "temperature": 0,
            "max_tokens": 10,
            "stream": True,
            "logprobs": 4
        }
        try:
            stream = client.completions.create(
                **request_content
            )
            finish_reason = None
            for completion in stream:
                # print(completion)
                # 必须参数校验
                for attr in ["id", "object", "model", "choices", "usage"]:
                    assert hasattr(completion, attr), f'completion返回参数校验失败'
                assert completion.model == model
                # 校验choices
                if finish_reason == None:
                    assert len(completion.choices) > 0
                for choice in completion.choices:
                    for attr in ["text", "index", "finish_reason", "logprobs"]:
                        assert hasattr(choice, attr), f'choice返回参数校验失败'
                    finish_reason = choice.finish_reason
                    if finish_reason is not None:
                        break

                    # 校验logprobs
                    assert len(choice.logprobs.tokens) == 1
                    target = choice.logprobs.tokens[0]
                    dic = choice.logprobs.top_logprobs[0]
                    max_prob_token = max(dic.keys(), key=lambda x: dic[x])
                    assert target == max_prob_token, f'校验logprobs失败:{target} != {max_prob_token}'

            assert finish_reason in ["stop", "length"]
        except Exception as e:
            logger.error(e)
            exc_info = sys.exc_info()
            logger.error("".join(traceback.format_exception(*exc_info)))
            pytest.fail()

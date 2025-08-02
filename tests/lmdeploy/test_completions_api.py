import sys
import traceback
import pytest
from ...utils.correctness_tool import CorrectnessTool
from ...utils.logger import logger


# Ordered by official OpenAI API documentation
# https://platform.openai.com/docs/api-reference/completions/create
class TestV1Completions:
    
    @pytest.mark.P2
    @pytest.mark.description("Completions接口对话推理测试: 非Stream模式 (低优, openai官方废弃)")
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
    @pytest.mark.description("Completions接口对话推理测试: Stream模式 (低优, openai官方废弃)")
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
            

from typing import Dict, Any, List

import openai
from openai import APIStatusError, Stream
from openai.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion import Choice, ChoiceLogprobs
from openai.types.completion_usage import CompletionUsage
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta


class OpenAIResponseValidator():
    def __init__(self, **params):
        self.request_content: Dict[str, Any] = params
        self.content: List[str] = [""] * self.request_content.get('n', 1)
        self.usage: List[CompletionUsage] = []
        self.chunk: List[ChatCompletionChunk] = []
        self.logprobs: List[ChoiceLogprobs] = []
                
        
    def __repr__(self):
        return (f"OpenAIResponseValidator(request_content={self.request_content!r}, "
                f"content={self.content!r}, usage={self.usage!r}, chunk={self.chunk!r})")


    def _validate_non_stream_response_structure(self, response: ChatCompletion, **kwargs) -> None:
        n = self.request_content.get('n', 1)
        # Check main attributes of ChatCompletion
        expected_attributes = ["id", "object", "created", "model", "choices", "usage"]
        for attr in expected_attributes:
            assert hasattr(response, attr), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
        assert isinstance(response.choices, list), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
        assert len(response.choices) == n, f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
        # Check model
        # assert response.model == self.request_content["model"], f'Return parameter validation failed, request ID: {response.id if response.id else -1}'

        # Check usage
        assert isinstance(response.usage, CompletionUsage), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
        self.usage.append(response.usage)

        # Check usage attributes
        usage_attributes = ["prompt_tokens", "completion_tokens", "total_tokens"]
        for attr in usage_attributes:
            assert hasattr(response.usage, attr), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
        if self.request_content.get("max_tokens", None):
            assert response.usage.completion_tokens <= self.request_content["max_tokens"] * n, \
                f'completion_tokens greater than max_tokens, request ID: {response.id if response.id else -1}'

        # Validate token count
        assert response.usage.completion_tokens > 0 and response.usage.prompt_tokens > 0, \
            f'Validation of completion_tokens>0 failed, request ID: {response.id if response.id else -1}'
        assert response.usage.total_tokens == response.usage.completion_tokens + response.usage.prompt_tokens, \
            f'Validation of total_tokens failed, request ID: {response.id if response.id else -1}'
            
        # Check choice
        for item in response.choices:
            assert item, f'choices is empty'
            # Check choice attributes
            choice_attributes = ["index", "message", "finish_reason"]
            for attr in choice_attributes:
                assert hasattr(item, attr), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'

            stop_reason = ["stop", "length"]
            assert item.finish_reason in stop_reason, f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
            assert isinstance(item.message, ChatCompletionMessage), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
            # Check message attributes
            message_attributes = ["role", "content"]
            for attr in message_attributes:
                assert hasattr(item.message, attr), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
            assert len(item.message.content) > 0, f'content is empty'
            self.content[item.index] = item.message.content

            # Check logprobs
            if self.request_content.get("logprobs", False):
                assert item.logprobs, f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
                assert hasattr(item.logprobs, "content"), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
                self.logprobs.append(item.logprobs)
                logprobs_content = item.logprobs.content
                assert len(logprobs_content) == response.usage.completion_tokens, "logprobs.content not equal to completion_tokens"
                for logprob in logprobs_content:
                    for attr in ["token", "logprob", "bytes"]:
                        assert hasattr(logprob, attr), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
                    if self.request_content.get("top_logprobs", None):
                        assert hasattr(logprob, "top_logprobs"), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
                        assert len(logprob.top_logprobs) == self.request_content["top_logprobs"], f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
                        for top_logprob in logprob.top_logprobs:
                            for attr in ["token", "logprob", "bytes"]:
                                assert hasattr(top_logprob, attr), f'Return parameter validation failed, request ID: {response.id if response.id else -1}'
        

    def _validate_stream_response_structure(self, response: Stream, **params) -> None:
        n = self.request_content.get("n", 1)
        finish_flag = False
        sub_finish_flag = [False] * n

        for chunk in response:
            print(chunk, flush=True)
            self.chunk.append(chunk)
            # Check main attributes of ChatCompletionChunk
            assert isinstance(chunk, ChatCompletionChunk), f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
            expected_attributes = ["id", "object", "created", "model", "choices"]
            for attr in expected_attributes:
                assert hasattr(chunk, attr), f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
            # Check model
            # assert chunk.model == self.request_content["model"], f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'

            # Check choices
            assert isinstance(chunk.choices, list), f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'

            if finish_flag == False:
                assert len(chunk.choices) > 0, f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
                choice = chunk.choices[0]
                # Validate required choice attributes
                choice_attributes = ["index", "delta", "finish_reason"]
                for attr in choice_attributes:
                    assert hasattr(choice, attr), f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
                # Validate index
                assert isinstance(choice.index, int) and choice.index<n, f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
                # Validate finish_reason
                if choice.finish_reason:
                    sub_finish_flag[choice.index] = True
                    assert choice.finish_reason in ["stop", "length"], f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
                # Validate delta
                assert isinstance(choice.delta, ChoiceDelta), f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
                delta_attributes = ["role", "content"]
                for attr in delta_attributes:
                    assert hasattr(choice.delta, attr), f'Return parameter validation failed, request ID: {chunk.id if chunk.id else -1}'
                # assert choice.delta.content, f'content is none, request ID: {chunk.id if chunk.id else -1}' # Allow individual chunk content to be empty, validate accumulated content at the end
                self.content[choice.index] += choice.delta.content if choice.delta.content is not None else ""
                    
            if all(sub_finish_flag):
                finish_flag = True

        assert finish_flag, "Stream mode did not return end flag"
        
        # Stream has ended, validate content
        for content in self.content:
            assert len(content) > 0, 'Stream mode content is empty'

        # Stream has ended, validate final usage
        if self.request_content.get("stream_options", {}).get("include_usage", False):
            last_chunk = self.chunk[-1]
            # Validate required usage attributes
            assert hasattr(last_chunk, "usage"), f'Return parameter validation failed, request ID: {last_chunk.id if last_chunk.id else -1}'
            usage_attributes = ["prompt_tokens", "completion_tokens", "total_tokens"]
            for attr in usage_attributes:
                assert hasattr(last_chunk.usage, attr), f'Return parameter validation failed, request ID: {last_chunk.id if last_chunk.id else -1}'
            # Validate token count
            if self.request_content.get("max_tokens", None):
                assert last_chunk.usage.completion_tokens <= self.request_content["max_tokens"] * n, \
                    f'completion_tokens greater than max_tokens, request ID: {last_chunk.id if last_chunk.id else -1}'
            assert last_chunk.usage.completion_tokens > 0 and last_chunk.usage.prompt_tokens > 0, \
                f'Validation of completion_tokens>0 failed, request ID: {last_chunk.id if last_chunk.id else -1}'
            assert last_chunk.usage.total_tokens == last_chunk.usage.completion_tokens + last_chunk.usage.prompt_tokens, \
                f'Validation of total_tokens failed, request ID: {last_chunk.id if last_chunk.id else -1}'
            self.usage.append(last_chunk.usage)
            assert len(self.usage) == 1, 'Stream mode usage is empty'
                    

    def _validate_api_error(self, e: Exception, use_docker: bool=False, error_code: int=None) -> None:
        assert isinstance(e, APIStatusError)
        assert e.status_code == 400
        assert len(e.body["message"]) > 0
        assert e.body["object"] == "error"
        
        if use_docker:
            assert e.body["code"] == error_code
        else:
            assert e.body["code"] == 400


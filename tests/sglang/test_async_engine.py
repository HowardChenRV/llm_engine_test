# test module
from llm.serving.engine.arg_utils import AsyncEngineArgs, EngineArgs
from llm.serving.engine.async_engine import AsyncEngine as AsyncLLMEngine
from llm.serving.sampling_params import SamplingParams
from llm.serving.tokenizer import get_tokenizer
from llm.serving.utils import random_uuid
from llm.serving.config import ScheduleStrategy, CostModelType
# other
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from dataclasses import fields
import argparse
import pytest
import os
# utils
from ...utils.logger import logger
from ...utils.nvidia_tool import set_cuda_visible_devices


class TestAyncEngine:
    
    # Test command line arguments for asynchronous inference: Validate default parameter values
    @pytest.mark.P1
    def test_add_async_engine_cli_args_default(self):
        parser = argparse.ArgumentParser()
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args([])  # Use an empty list to simulate not passing any command line arguments

        assert hasattr(args, "model")
        assert args.model == "facebook/opt-125m"

        assert hasattr(args, "tokenizer")
        assert args.tokenizer == EngineArgs.tokenizer

        assert hasattr(args, "trust_remote_code")
        assert args.trust_remote_code is False

        assert hasattr(args, "max_model_len")
        assert args.max_model_len is None

        assert hasattr(args, "max_gpu_blocks")
        assert args.max_gpu_blocks == 0

        assert hasattr(args, "endpoints")
        assert args.endpoints == []

        assert hasattr(args, "pipeline_parallel_size")
        assert args.pipeline_parallel_size == EngineArgs.pipeline_parallel_size

        assert hasattr(args, "tensor_parallel_size")
        assert args.tensor_parallel_size == EngineArgs.tensor_parallel_size

        assert hasattr(args, "block_size")
        assert args.block_size == EngineArgs.block_size

        assert hasattr(args, "enable_prefix_caching")
        assert args.enable_prefix_caching is False

        assert hasattr(args, "seed")
        assert args.seed == EngineArgs.seed

        assert hasattr(args, "swap_space")
        assert args.swap_space == EngineArgs.swap_space

        assert hasattr(args, "gpu_memory_utilization")
        assert args.gpu_memory_utilization == EngineArgs.gpu_memory_utilization

        assert hasattr(args, "max_num_batched_tokens")
        assert args.max_num_batched_tokens == EngineArgs.max_num_batched_tokens

        assert hasattr(args, "max_num_seqs")
        assert args.max_num_seqs == EngineArgs.max_num_seqs

        assert hasattr(args, "max_paddings")
        assert args.max_paddings == EngineArgs.max_paddings

        assert hasattr(args, "quantization")
        assert args.quantization is None

        assert hasattr(args, "backend")
        assert args.backend == EngineArgs.backend

        assert hasattr(args, "prefill_strategy")
        assert args.prefill_strategy == EngineArgs.prefill_strategy

        assert hasattr(args, "spec_decoding_method")
        assert args.spec_decoding_method == EngineArgs.spec_decoding_method

        assert hasattr(args, "spec_decoding_params")
        assert args.spec_decoding_params is None

        assert hasattr(args, "disable_kv_fusion")
        assert args.disable_kv_fusion is False

        assert hasattr(args, "disable_log_stats")
        assert args.disable_log_stats is False

        assert hasattr(args, "schedule_strategy")
        assert args.schedule_strategy == ScheduleStrategy.DEFAULT

        assert hasattr(args, "cost_model_type")
        assert args.cost_model_type == CostModelType.DEFAULT

        assert hasattr(args, "slo_ttft")
        assert args.slo_ttft == EngineArgs.slo_ttft

        assert hasattr(args, "slo_tpot")
        assert args.slo_tpot == EngineArgs.slo_tpot

        assert hasattr(args, "enable_peak_memory_predict")
        assert args.enable_peak_memory_predict is False
        
        assert hasattr(args, "disable_log_requests")
        assert args.disable_log_requests is False

        assert hasattr(args, "max_log_len")
        assert args.max_log_len is None


    # Validate default values of asynchronous inference parameter objects
    @pytest.mark.P1
    def test_async_engine_args_default(self):
        parser = argparse.ArgumentParser()
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args([])  # Use an empty list to simulate not passing any command line arguments
        engine_args = AsyncEngineArgs.from_cli_args(args)
        
        # Define expected default values
        expected_defaults = {
            "model": "facebook/opt-125m",
            "tokenizer": "facebook/opt-125m",
            "trust_remote_code": False,
            "seed": 0,
            "pipeline_parallel_size": 1,
            "tensor_parallel_size": 1,
            "block_size": 16,
            "enable_prefix_caching": False,
            "swap_space": 0,
            "gpu_memory_utilization": 0.90,
            "max_num_batched_tokens": 4096,
            "max_num_seqs": 256,
            "max_paddings": 256,
            "quantization": None,
            "endpoints": [],
            "backend": "rpyc",
            "max_gpu_blocks": 0,
            "prefill_strategy": "varlen",
            "spec_decoding_method": None,
            "spec_decoding_params": None,
            "disable_kv_fusion": False,
            "disable_log_stats": False,
            "schedule_strategy": ScheduleStrategy.DEFAULT,
            "cost_model_type": CostModelType.DEFAULT,
            "slo_ttft": 10000,
            "slo_tpot": 10000,
            "enable_peak_memory_predict": False
        }
        
        for field in fields(EngineArgs):
            field_name = field.name
            assert getattr(engine_args, field_name) == expected_defaults[field_name], \
                f"Default value for {field_name} is not as expected."

    
    # Test tokenizer acquisition: Different models acquire tokenizer normally
    @pytest.mark.P0
    @pytest.mark.parametrize("model_path", [
        "/share/datasets/public_models/Llama-2-70b-chat-hf",
        "/share/datasets/public_models/Meta-Llama-3-70B-Instruct-hf",
        "/share/datasets/public_models/Qwen_Qwen1.5-72B-Chat",
        "/share/datasets/public_models/Qwen_Qwen2-72B-Instruct"
    ])
    def test_get_tokenizer_default(self, model_path):
        llm_tokenizer = get_tokenizer(model_path, trust_remote_code=True)
        assert llm_tokenizer is not None
        if not isinstance(llm_tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
            logger.error(f"Unexpected tokenizer: {llm_tokenizer}")
            raise AssertionError(f"Unexpected tokenizer type returned for model {model_path}.") 
        logger.info(f"tokenizer pass: {model_path}")  


    # Test uuid: Whether the length is 32, whether the format is hexadecimal, and whether multiple acquisitions are inconsistent
    @pytest.mark.P0
    def test_random_uuid(self):
        request_id = random_uuid()
        assert len(request_id) == 32
        assert all(c in "0123456789abcdef" for c in request_id)
        request_id2 = random_uuid()
        assert request_id != request_id2


    # Test SamplingParams normal settings
    @pytest.mark.P0
    def test_sampling_params_set(self):
        test_params = {
            "n": 2,
            "best_of": 2,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "repetition_penalty": 0.5,
            "temperature": 0.8,
            "top_p": 0.5,
            "top_k": 1,
            "min_p": 0.1,
            # "use_beam_search": False,
            # "length_penalty": 0.1,
            # "early_stopping": True,
            "stop": [1, 2],
            "stop_token_ids": [1, 2],
            "include_stop_str_in_output": True,
            "ignore_eos": True,
            "max_tokens": 32,
            "input_token_length": 32,
            "logprobs": 1,
            "prompt_logprobs": 1,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False
        }
        sampling_params = SamplingParams(**test_params)
        attributes = sampling_params.__dict__
        for key, value in test_params.items():
            assert attributes[key] == value, f"Attribute {key} expected {value} but got {attributes[key]}"


    # Test SamplingParams Greedy settings
    @pytest.mark.P1
    def test_sampling_params_set_by_greedy(self):
        sampling_params = SamplingParams(
            temperature = 0.0,
            top_p = 0.5,
            top_k = 1,
            min_p = 0.1,
            best_of = 1
        )
        assert sampling_params.temperature == 0.0
        assert sampling_params.top_p == 1.0
        assert sampling_params.top_k == -1
        assert sampling_params.min_p == 0.0


    # Test SamplingParams Greedy settings
    @pytest.mark.P2
    def test_sampling_params_set_by_greedy_best_of_2(self):
        with pytest.raises(ValueError):
            sampling_params = SamplingParams(
                temperature = 0.0,
                best_of = 2
            )
            logger.debug(sampling_params)
        

    # Test asynchronous inference interface
    @pytest.mark.P0
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", [
        {"tp": 1, "model_path": "/share/datasets/public_models/Meta-Llama-3-8B-Instruct"},
    ])
    async def test_async_engine_inference(self, model):
        # Process test parameters
        tp = model["tp"]
        model_path = model["model_path"]
        # Set CUDA_VISIBLE_DEVICES
        set_cuda_visible_devices(gpu_num=tp)
        # Set llm log level
        os.environ["INFI_LOG_LEVEL"] = "3"
        
        args_list = [
            "--host", "0.0.0.0",
            "--model-path", model_path,
            "--tensor-parallel-size", str(tp),
            "--trust-remote-code",
            "--gpu-memory-utilization", "0.95"
        ]
        
        parser = self.get_fastchat_parser()
        parser = AsyncEngineArgs.add_cli_args(parser)
        args = parser.parse_args(args_list)
        if args.model_path:
            args.model = args.model_path
        
        # 创建引擎
        engine_args = AsyncEngineArgs.from_cli_args(args)
        engine = AsyncLLMEngine.from_engine_args(engine_args)

        # Inference
        async def inference_generator(engine, max_tokens: int, boundary_len: int):
            prompt = "The biggest animal in the world is"
            request_id = random_uuid()
            sampling_params = SamplingParams(
                temperature = 0.0,
                max_tokens = max_tokens
            )
            results_generator = engine.generate(prompt, sampling_params, request_id)
        
            final_output = None
            count = 0
            async for request_output in results_generator:
                count += 1
                final_output = request_output
                # Truncate when reaching the maximum inference count, test abort method
                if count >= boundary_len:
                    await engine.abort(request_id)
            
            logger.info(f"inference_generator: {final_output}")
            # Validate
            assert final_output.request_id == request_id
            assert final_output.prompt == prompt
            assert len(final_output.outputs) > 0
            if boundary_len < max_tokens:
                assert final_output.finished == False
            else:
                assert final_output.finished == True
            
            return final_output
        
        response = await inference_generator(engine=engine, max_tokens=128, boundary_len=1024)
        response_abort = await inference_generator(engine=engine, max_tokens=128, boundary_len=32)

        # Validate
        output = response.outputs[0]
        output_abort = response_abort.outputs[0]
        assert output.index == 0
        assert "blue whale" in output.text
        assert len(output.token_ids) == 128
        assert len(output_abort.token_ids) == 32        
    
    
    @classmethod
    def get_fastchat_parser(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--host", type=str, default="localhost")
        parser.add_argument("--port", type=int, default=21002)
        parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
        parser.add_argument(
            "--controller-address", type=str, default="http://localhost:21001"
        )
        parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")
        parser.add_argument(
            "--model-names",
            type=lambda s: s.split(","),
            help="Optional display comma separated names",
        )
        parser.add_argument("--limit-worker-concurrency", type=int, default=1024)
        parser.add_argument("--no-register", action="store_true")
        parser.add_argument("--num-gpus", type=int, default=1)
        parser.add_argument(
            "--conv-template", type=str, default=None, help="Conversation prompt template."
        )
        parser.add_argument(
            "--trust_remote_code",
            action="store_false",
            default=True,
            help="Trust remote code (e.g., from HuggingFace) when"
            "downloading the model and tokenizer.",
        )
        parser.add_argument(
            "--gpu_memory_utilization",
            type=float,
            default=0.9,
            help="The ratio (between 0 and 1) of GPU memory to"
            "reserve for the model weights, activations, and KV cache. Higher"
            "values will increase the KV cache size and thus improve the model's"
            "throughput. However, if the value is too high, it may cause out-of-"
            "memory (OOM) errors.",
        )
        
        return parser



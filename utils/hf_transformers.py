from transformers import pipeline
import torch


def run_correct_hf(model_path, messages, result):
    pipe = pipeline(
        "text-generation",
        model=model_path,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device="cuda",
    )

    outputs = pipe(
        messages,
        max_new_tokens=50,
    )
    for i, item in enumerate(outputs):
        result[i]["hf"] = item[0]["generated_text"][len(messages[i]):]

    print(result)
    return

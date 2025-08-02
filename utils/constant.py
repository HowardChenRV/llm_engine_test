# Inference engine feature-related
Feature_config = {
    "prefix_caching": {
        "tp": 2,
        "model_path": "/share/datasets/tmp_share/chenyonghua/models/Vllm_AWQ_models/qwen1.5/Qwen_Qwen1.5-72B-Chat_AWQ",
        "cmd_suffix": ["--enable-prefix-caching"],
    },
    "peak_memory_predict": {
        "tp": 1,
        "model_path": "/share/datasets/public_models/Meta-Llama-3-8B-Instruct",
        "cmd_suffix": ["--max-gpu-blocks", "1000", "--enable-peak-memory-predict"],
    },
    "spec_decoding": {
        "tp": 1,
        "model_path": "/share/datasets/public_models/Llama-2-7b-hf",
        "cmd_suffix": ["--max-num-seqs", "4", "--spec-decoding-method", "LADE", "--spec-decoding-params", "W=3", "N=5", "G=3", "pool_from_prompt=True"]
    },
    "dynamic_strategy": {
        "tp": 1,
        "model_path": "/share/datasets/public_models/Meta-Llama-3-8B-Instruct",
        "cmd_suffix": ["--schedule-strategy", "DYNAMIC", "--slo-ttft", "1000", "--slo-tpot", "100"]
    }
}


# Self-built 10 questions to test inference engine correctness
Question_10 = [
    "what is the largest animal in the world?",
    "What is the square root of 81?",
    "Who was president of the United States during the Civil War?",
    "Are leaves falling in the wind free fall?",
    "What are the four major ancient civilizations in the world?",
    "A says: I can swim; B says: A cannot swim; C says: B cannot swim; D says: Three of us can swim. Only one person is lying, who is it?",
    "Six students volunteer at three venues: A, B, and C. Each student goes to only one venue. Venue A arranges for 1 student, venue B arranges for 2 students, and venue C arranges for 3 students. How many different arrangements are there? Please provide only the final numerical result.",
    "There is a circular track with a circumference of 600 meters. A and B start running simultaneously from the same place in the same direction. A runs 300 meters per minute, and B runs 400 meters per minute. After how many minutes will they meet for the first time?",
    "Given an arithmetic sequence: 5, 9, 13, 17...137, which term is 81 in this sequence?",
    "Teacher Wang has a box of pencils. When distributed equally among 2 students, there is 1 pencil left over. When distributed equally among 3 students, there are 2 pencils left over. When distributed equally among 4 students, there are 3 pencils left over. When distributed equally among 5 students, there are 4 pencils left over. What is the minimum number of pencils in the box?",
]

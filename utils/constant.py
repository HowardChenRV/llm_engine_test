# 推理引擎特性功能相关
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


# 自建10题 测试推理引擎正确性
Question_10 = [
    "what is the largest animal in the world?",
    "What is the square root of 81?",
    "Who was president of the United States during the Civil War?",
    "Are leaves falling in the wind free fall?",
    "What are the four major ancient civilizations in the world?",
    "甲说：我会游泳；乙说：甲不会游泳；丙说：乙不会游泳；丁说：我们有三个人会游泳。以上只有一个人说假话，那么谁说假话？",
    "6名同学到甲、乙、丙三个场馆做志愿者，每名同学只去1个场馆，甲场馆安排1名，乙场馆安排2名，丙场馆安排3名，则不同的安排方法共有多少种？只回答最后的结果数字",
    "有一周长600米的环形跑道，甲、乙二人同时、同地、同向而行，甲每分钟跑300米，乙每分钟跑400米，经过几分钟二人第一次相遇？",
    "已知等差数列5，9，13，17...137，问81是这个数列的第几个数",
    "王老师有一盒铅笔，如平均分给2名同学余1支，平均分给3名同学余2支，平均分给4名同学余3支，平均分给5名同学余4支。问这盒铅笔最少有多少支？",
]

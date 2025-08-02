

# 预留: 先简单校验，后续扩展
class CorrectnessTool:
    
    QUESTION_LIST = [
        {  
            "idx": 0,
            "prompt": "The biggest animal in the world is",
            "answer": "blue whale",
            "max_tokens": 4096
        }
    ]
    
    @classmethod
    def get_question(cls):
        return cls.QUESTION_LIST[0]
    
    @classmethod
    def check_answer(cls, answer: str):
        except_answer = cls.get_question()["answer"]
        assert except_answer in answer.lower(), f"Check answer error! answer: {answer}"
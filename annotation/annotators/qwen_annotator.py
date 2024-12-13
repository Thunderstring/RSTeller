from .base_annotator import BaseAnnotator

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class QwenAnnotator(BaseAnnotator):
    
    def __init__(self, model_id, **kwargs):
        super().__init__( **kwargs)
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
        # max_tokens is for the maximum length for generation.
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)

        # Input the model name or path. Can be GPTQ or AWQ models.
        self.llm = LLM(model="model_id")
        
    def inference(self, text):
        
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": text}
        ]
                
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        outputs = self.llm.generate([text], self.sampling_params, use_tqdm=False)
        
        return outputs[0].outputs[0].text
        
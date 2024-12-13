from .base_annotator import BaseAnnotator

from vllm import LLM, SamplingParams

class MixtralAnnotator(BaseAnnotator):
    
    def __init__(self, model_id, **kwargs):
        super().__init__( **kwargs)
        self.llm = LLM(model=model_id, tokenizer_mode='mistral', config_format='mistral', load_format='mistral')
        self.sampling_params = SamplingParams(max_tokens=300)
        
    def inference(self, text):
        
        messages = [
            {
                "role": "user",
                "content": text
            },
        ]
        
        outputs = self.llm.chat(messages, sampling_params=self.sampling_params, use_tqdm=False)
        
        return outputs[0].outputs[0].text
        

# class Mixtral_7B_v0_3(BaseMixtralAnnotator):
    
#     def __init__(self, **kwargs):
#         super().__init__('mistralai/Mistral-7B-v0.3', **kwargs)
#         self.annotator_id = 3
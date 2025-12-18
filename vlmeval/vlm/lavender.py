import torch
from .llama_vision import llama_vision
from peft import PeftModel


class llama_vision_lavender(llama_vision):
    """Lavender LoRA"""
    
    INSTALL_REQ = False
    INTERLEAVE = False
    
    def __init__(self, 
                 model_path='meta-llama/Llama-3.2-11B-Vision-Instruct',
                 lora_path='lxasqjc/lavender-llama-3.2-11b-lora',
                 **kwargs):
        super().__init__(model_path=model_path, **kwargs)
        
        print(f'Loading Lavender LoRA from {lora_path}...')
        self.model = PeftModel.from_pretrained(
            self.model,
            lora_path,
            torch_dtype=torch.bfloat16
        ).eval()
        
        self.model_name = f'{model_path}+{lora_path}'

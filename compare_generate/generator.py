"""
简单的代码生成器
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from typing import Optional


class CodeGenerator:
    def __init__(self, model_path: str):
        """初始化生成器"""
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model_name = model_path.split('/')[-1]
        print(f"✓ Model loaded: {self.model_name}")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """生成代码"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    def extract_code(self, text: str) -> Optional[str]:
        """提取代码块"""
        # 提取 ```python ... ``` 之间的代码
        pattern = r'```python\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # 提取 ``` ... ``` 之间的代码
        pattern = r'```\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        return text

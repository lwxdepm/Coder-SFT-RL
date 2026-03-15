"""
Code Generator - 模型加载和生成
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class CodeGenerator:
    def __init__(self, model_path):
        """加载模型和tokenizer"""
        print(f"Loading model from {model_path}...")
        
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✓ Model loaded: {model_path.split('/')[-1]}")
    
    def generate(self, prompt, temperature=0.7, max_tokens=512, top_p=0.95):
        """生成代码"""
        # 构建消息格式（适用于 Instruct 模型）
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 使用 chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = f"User: {prompt}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,  # 必须为 True 才能用 temperature
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码（只取新生成的部分）
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()


# 测试代码
if __name__ == "__main__":
    # 测试单个模型
    gen = CodeGenerator("Qwen/Qwen2.5-Coder-1.5B-Instruct")
    result = gen.generate("Write a Python function to check if a number is prime.")
    print(result)

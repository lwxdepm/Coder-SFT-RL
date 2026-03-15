"""
Test model (both original and trained models)
"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List


def load_model(model_path):
    """加载模型"""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    return model, tokenizer


def generate_code(model, tokenizer, prompt, **kwargs):
    """生成代码"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=kwargs.get('max_new_tokens', 256),
            temperature=kwargs.get('temperature', 0.7),
            do_sample=kwargs.get('do_sample', True),
            top_p=kwargs.get('top_p', 0.95),
            num_return_sequences=kwargs.get('num_return_sequences', 1),
        )
    
    results = []
    for output in outputs:
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        results.append(generated_text)
    
    return results


def get_test_prompts() -> List[str]:
    """获取测试 prompts"""
    return [
        """Write a function named average_goals that takes a list of integers 'goals', where each integer represents the number of goals scored by a football team in a match. The function should return a float rounded to two decimal places representing the average number of goals scored per match. If the list is empty, the function should return 0.00.\n\nExample:\ngoals = [2, 0, 3, 1, 4]\nprint(average_goals(goals))  # Output: 2.00
""",
        """You are tasked with creating a Python function that takes a list of integers and returns a new list containing only the even numbers from the input list. The function should maintain the order of the original list. Implement the function `filter_even_numbers(nums: List[int]) -> List[int]` where `nums` is a list of integers. Your function should return a list of even integers derived from `nums`. If there are no even integers, return an empty list.
"""
    ]


def test_single_model(model_path, num_samples=1, **gen_kwargs):
    """测试单个模型"""
    model, tokenizer = load_model(model_path)
    
    test_prompts = get_test_prompts()
    
    print("\n" + "="*80)
    print(f"Testing model: {model_path}")
    print("="*80)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(test_prompts)}")
        print(f"{'='*80}")
        print(f"\nPrompt:")
        print(prompt)
        print(f"\nGenerated Code:")
        print("-"*80)
        
        results = generate_code(
            model, 
            tokenizer, 
            prompt, 
            num_return_sequences=num_samples,
            **gen_kwargs
        )
        
        for j, result in enumerate(results, 1):
            if num_samples > 1:
                print(f"\n--- Sample {j} ---")
            print(result)
            if j < len(results):
                print("-"*80)
        
        print("-"*80)
    
    print("\n" + "="*80)


def compare_models(base_model_path, trained_model_path, num_samples=1, **gen_kwargs):
    """对比两个模型"""
    print("\n" + "="*80)
    print("COMPARING MODELS")
    print("="*80)
    print(f"Base model:    {base_model_path}")
    print(f"Trained model: {trained_model_path}")
    print("="*80)
    
    base_model, base_tokenizer = load_model(base_model_path)
    trained_model, trained_tokenizer = load_model(trained_model_path)
    
    test_prompts = get_test_prompts()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}/{len(test_prompts)}")
        print(f"{'='*80}")
        print(f"\nPrompt:")
        print(prompt)
        
        # 基础模型生成
        print(f"\n{'─'*80}")
        print(f"BASE MODEL OUTPUT:")
        print(f"{'─'*80}")
        base_results = generate_code(
            base_model, 
            base_tokenizer, 
            prompt,
            num_return_sequences=num_samples,
            **gen_kwargs
        )
        for j, result in enumerate(base_results, 1):
            if num_samples > 1:
                print(f"\n--- Sample {j} ---")
            print(result)
            if j < len(base_results):
                print("-"*40)
        
        # 训练后模型生成
        print(f"\n{'─'*80}")
        print(f"TRAINED MODEL OUTPUT:")
        print(f"{'─'*80}")
        trained_results = generate_code(
            trained_model, 
            trained_tokenizer, 
            prompt,
            num_return_sequences=num_samples,
            **gen_kwargs
        )
        for j, result in enumerate(trained_results, 1):
            if num_samples > 1:
                print(f"\n--- Sample {j} ---")
            print(result)
            if j < len(trained_results):
                print("-"*40)
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Test language models")
    parser.add_argument("--model_path", type=str, help="Path to model to test")
    parser.add_argument("--base_model", type=str, help="Path to base model (for comparison)")
    parser.add_argument("--trained_model", type=str, help="Path to trained model (for comparison)")
    parser.add_argument("--compare", action="store_true", help="Compare base and trained models")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--max_new_tokens", type=int, default=4000, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling")
    
    args = parser.parse_args()
    
    gen_kwargs = {
        'max_new_tokens': args.max_new_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'do_sample': True if args.temperature > 0 else False,
    }
    
    if args.compare:
        if not args.base_model or not args.trained_model:
            parser.error("--compare requires both --base_model and --trained_model")
        compare_models(args.base_model, args.trained_model, args.num_samples, **gen_kwargs)
    elif args.model_path:
        test_single_model(args.model_path, args.num_samples, **gen_kwargs)
    else:
        parser.error("Either --model_path or --compare (with --base_model and --trained_model) is required")


if __name__ == "__main__":
    main()
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "saves/qwen1.5b_native_full"

print("🚀 正在加载专属代码大模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cuda"
)

print("\n💡 模型加载完毕！输入 'quit' 退出，输入 'clear' 清空上下文。")
messages = [{"role": "system", "content": "You are a helpful and expert programming assistant."}]

while True:
    user_input = input("\n🧑‍💻 你: ")
    if user_input.lower() == 'quit':
        break
    if user_input.lower() == 'clear':
        messages = [{"role": "system", "content": "You are a helpful and expert programming assistant."}]
        print("🧹 上下文已清空")
        continue

    messages.append({"role": "user", "content": user_input})
    
    # 应用对话模板
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print("🤖 模型思考中...")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,      # ✅ 开启采样模式！
            temperature=0.3,     # 现在这个参数会生效了
            top_p=0.9,           # 现在这个参数也会生效了
            repetition_penalty=1.05
        )

    # 截取新生成的回复
    response_ids = generated_ids[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    print(f"\n✨ 模型: {response}")
    messages.append({"role": "assistant", "content": response})
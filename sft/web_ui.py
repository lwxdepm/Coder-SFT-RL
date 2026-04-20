import gradio as gr
from openai import OpenAI

# 1. 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-local-vllm"
)

MODEL_NAME = "saves/qwen1.5b_native_full"

def predict(message, history):
    """与 vLLM 交互的核心函数"""
    messages = [{"role": "system", "content": "You are a helpful and expert programming assistant."}]
    
    # ✅ 完美适配 Gradio 5.x 的新版 history 格式
    for msg in history:
        # 提取角色和内容（兼容字典和对象两种可能的返回格式）
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg["content"] if isinstance(msg, dict) else msg.content
        messages.append({"role": role, "content": content})
        
    # 加入当前用户的最新提问
    messages.append({"role": "user", "content": message})

    # 发起流式 API 请求
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3, 
        stream=True,      
        stop=["<|im_end|>"] # 留着防暴走双保险
    )

    # 捕获流式输出
    partial_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            partial_message += chunk.choices[0].delta.content
            yield partial_message

# 2. 构建并启动 Web 前端
print("🚀 正在启动 Web 交互界面...")
demo = gr.ChatInterface(
    predict,
    title="⚡ Qwen-1.5B 极速代码助手 (vLLM Powered)",
    description="基于本地 RTX 4090 部署，完美兼容 OpenAI API 格式，体验毫秒级延迟的代码生成！",
    examples=["用 Python 写一个支持多 Agent 协作的基类框架", "用 NumPy 实现快速傅里叶变换 (FFT)"]
)

if __name__ == "__main__":
    # ✅ 关闭 share=True，避免触发 AutoDL 的防火墙下载报错
    # 请直接使用 AutoDL 控制台的【自定义服务】按钮来访问
    demo.launch(server_name="0.0.0.0", server_port=6006, share=False)
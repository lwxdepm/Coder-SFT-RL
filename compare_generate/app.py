"""
代码生成对比界面 - 兼容 Gradio 6.0
"""
import gradio as gr
from generator import CodeGenerator
import os


# 预设的示例问题
EXAMPLES = [
    "Write a Python function that filters even numbers from a list of integers.",
    "Write a function to calculate the nth Fibonacci number.",
    "Write a function to reverse a string.",
    "Write a function to check if a string is a palindrome.",
    "Write a function to find the maximum element in a list.",
]


def load_models():
    """加载模型"""
    base_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    trained_path = "../exported_models/grpo_merged"
    
    # 检查路径
    if not os.path.exists(trained_path):
        print(f"Warning: Trained model not found at {trained_path}")
        print("Using base model for both...")
        trained_path = base_path
    
    base_gen = CodeGenerator(base_path)
    trained_gen = CodeGenerator(trained_path)
    
    return base_gen, trained_gen


# 加载模型
print("Initializing models...")
base_generator, trained_generator = load_models()
print("✓ All models loaded!\n")


def generate_single(prompt, temperature, max_tokens, model_choice):
    """单模型生成"""
    if not prompt.strip():
        return "❌ Please enter a prompt"
    
    generator = trained_generator if model_choice == "Trained Model" else base_generator
    
    try:
        output = generator.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def generate_compare(prompt, temperature, max_tokens):
    """对比生成"""
    if not prompt.strip():
        return "❌ Please enter a prompt", "❌ Please enter a prompt"
    
    try:
        # 生成基础模型输出
        base_output = base_generator.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 生成训练模型输出
        trained_output = trained_generator.generate(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return base_output, trained_output
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg


# 创建 Gradio 界面
with gr.Blocks(title="Code Generator") as demo:
    
    gr.Markdown("# 🤖 Code Generator")
    gr.Markdown("Generate code with AI models - Single or Compare mode")
    
    with gr.Tabs():
        
        # Tab 1: 单模型生成
        with gr.Tab("Single Model"):
            with gr.Row():
                with gr.Column(scale=2):
                    single_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your coding problem...",
                        lines=5
                    )
                    
                    with gr.Row():
                        single_model = gr.Radio(
                            ["Base Model", "Trained Model"],
                            label="Model",
                            value="Trained Model"
                        )
                    
                    with gr.Row():
                        single_temp = gr.Slider(
                            0.1, 1.5, 
                            value=0.7, 
                            step=0.1, 
                            label="Temperature"
                        )
                        single_tokens = gr.Slider(
                            128, 1024, 
                            value=512, 
                            step=128, 
                            label="Max Tokens"
                        )
                    
                    single_btn = gr.Button("🚀 Generate", variant="primary")
                    
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=single_prompt,
                        label="Example Prompts"
                    )
                
                with gr.Column(scale=3):
                    single_output = gr.Textbox(
                        label="Generated Code",
                        lines=20
                    )
            
            single_btn.click(
                fn=generate_single,
                inputs=[single_prompt, single_temp, single_tokens, single_model],
                outputs=single_output
            )
        
        # Tab 2: 对比生成
        with gr.Tab("Compare Models"):
            with gr.Row():
                with gr.Column(scale=1):
                    compare_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your coding problem...",
                        lines=8
                    )
                    
                    with gr.Row():
                        compare_temp = gr.Slider(
                            0.1, 1.5, 
                            value=0.7, 
                            step=0.1, 
                            label="Temperature"
                        )
                        compare_tokens = gr.Slider(
                            128, 1024, 
                            value=512, 
                            step=128, 
                            label="Max Tokens"
                        )
                    
                    compare_btn = gr.Button("🔄 Compare", variant="primary")
                    
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=compare_prompt,
                        label="Example Prompts"
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Base Model")
                    base_output = gr.Textbox(
                        label="Output",
                        lines=20
                    )
                
                with gr.Column():
                    gr.Markdown("### ⭐ Trained Model")
                    trained_output = gr.Textbox(
                        label="Output",
                        lines=20
                    )
            
            compare_btn.click(
                fn=generate_compare,
                inputs=[compare_prompt, compare_temp, compare_tokens],
                outputs=[base_output, trained_output]
            )
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tips**: Lower temperature = more deterministic, Higher temperature = more creative")


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft()  # 移到这里
    )
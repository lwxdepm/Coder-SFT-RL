"""
代码生成对比界面 - Gradio 6.0
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
    "Write a function to sort a list using bubble sort.",
    "Write a function to count words in a sentence.",
    "Write a function to check if two strings are anagrams.",
]


def load_models():
    """加载模型"""
    base_path = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    trained_path = "../exported_models/grpo_merged"
    
    # 获取绝对路径
    abs_trained_path = os.path.abspath(trained_path)
    
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)
    
    # 检查训练模型是否存在
    if not os.path.exists(trained_path):
        print(f"⚠️  Trained model not found at: {abs_trained_path}")
        print(f"⚠️  Using base model for both (outputs will be similar!)")
        trained_path = base_path
    else:
        print(f"✓ Found trained model: {abs_trained_path}")
    
    print(f"\nBase model: {base_path}")
    print(f"Trained model: {trained_path}")
    print()
    
    # 加载模型
    base_gen = CodeGenerator(base_path)
    trained_gen = CodeGenerator(trained_path)
    
    # 验证是不同的对象
    assert base_gen is not trained_gen, "Same generator object!"
    assert base_gen.model is not trained_gen.model, "Same model object!"
    
    print("\n" + "=" * 60)
    print("✓ Both models loaded successfully!")
    print("=" * 60 + "\n")
    
    return base_gen, trained_gen


# 加载模型
print("Initializing models...")
base_generator, trained_generator = load_models()


def generate_single(prompt, temperature, max_tokens, model_choice):
    """单模型生成"""
    if not prompt.strip():
        return "❌ Please enter a prompt"
    
    generator = trained_generator if model_choice == "Trained Model" else base_generator
    
    try:
        output = generator.generate(
            prompt,
            temperature=temperature,
            max_tokens=int(max_tokens)
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
            max_tokens=int(max_tokens)
        )
        
        # 生成训练模型输出
        trained_output = trained_generator.generate(
            prompt,
            temperature=temperature,
            max_tokens=int(max_tokens)
        )
        
        return base_output, trained_output
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        return error_msg, error_msg


# 创建 Gradio 界面
with gr.Blocks(title="Code Generator", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("# 🤖 Code Generator - GRPO Training Demo")
    gr.Markdown("Compare outputs from base model vs GRPO-trained model")
    
    with gr.Tabs():
        
        # Tab 1: 对比生成（主要功能）
        with gr.Tab("🔄 Compare Models"):
            with gr.Row():
                with gr.Column(scale=1):
                    compare_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Enter your coding problem...",
                        lines=6
                    )
                    
                    with gr.Row():
                        compare_temp = gr.Slider(
                            0.1, 1.5, 
                            value=0.7, 
                            step=0.1, 
                            label="Temperature",
                            info="Higher = more creative"
                        )
                        compare_tokens = gr.Slider(
                            128, 1024, 
                            value=512, 
                            step=64, 
                            label="Max Tokens"
                        )
                    
                    compare_btn = gr.Button("🚀 Generate & Compare", variant="primary", size="lg")
                    
                    gr.Markdown("### 📝 Example Prompts")
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=compare_prompt,
                        label=""
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📦 Base Model (Qwen2.5-Coder-1.5B)")
                    base_output = gr.Textbox(
                        label="Output",
                        lines=20,
                        show_copy_button=True
                    )
                
                with gr.Column():
                    gr.Markdown("### ⭐ GRPO Trained Model")
                    trained_output = gr.Textbox(
                        label="Output",
                        lines=20,
                        show_copy_button=True
                    )
            
            compare_btn.click(
                fn=generate_compare,
                inputs=[compare_prompt, compare_temp, compare_tokens],
                outputs=[base_output, trained_output]
            )
        
        # Tab 2: 单模型生成
        with gr.Tab("📝 Single Model"):
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
                            label="Select Model",
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
                            step=64, 
                            label="Max Tokens"
                        )
                    
                    single_btn = gr.Button("🚀 Generate", variant="primary")
                    
                    gr.Examples(
                        examples=EXAMPLES[:4],
                        inputs=single_prompt,
                        label="Examples"
                    )
                
                with gr.Column(scale=3):
                    single_output = gr.Textbox(
                        label="Generated Code",
                        lines=20,
                        show_copy_button=True
                    )
            
            single_btn.click(
                fn=generate_single,
                inputs=[single_prompt, single_temp, single_tokens, single_model],
                outputs=single_output
            )
    
    gr.Markdown("---")
    gr.Markdown("""
    💡 **Tips**: 
    - Lower temperature (0.1-0.5) = more deterministic outputs
    - Higher temperature (0.7-1.2) = more creative/varied outputs
    - The trained model should produce more concise, correct code
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

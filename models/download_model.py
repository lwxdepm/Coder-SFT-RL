import os
from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()

def download_hf_model(repo_id: str, local_dir: str):
    """
    从 Hugging Face 下载整个模型仓库
    """
    # 针对国内网络环境，自动配置 HF 镜像源加速下载
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    console.print(f"[bold cyan]🚀 准备下载模型: {repo_id}[/bold cyan]")
    console.print(f"📂 目标本地路径: [green]{local_dir}[/green]")
    console.print("[yellow]💡 提示: 脚本已开启断点续传功能。如果网络中断，重新运行本脚本即可恢复进度。[/yellow]\n")
    
    try:
        # snapshot_download 会自动处理多线程下载和完整性校验
        download_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 在 WSL2 建议设为 False，避免 Windows 文件系统软链接报错
            resume_download=True,          # 开启断点续传
            max_workers=4                  # 并发下载线程数
        )
        console.print(f"\n[bold green]🎉 模型下载成功！已完整保存在:[/bold green] {download_path}")
        
    except Exception as e:
        console.print(f"\n[bold red]❌ 下载过程中出现错误:[/bold red] {str(e)}")

if __name__ == "__main__":
    # Hugging Face 上的模型仓库名称
    MODEL_REPO = "Qwen/Qwen2.5-Coder-1.5B"
    
    # 你想保存在本地的绝对或相对路径
    # 建议统一放在一个 models 文件夹下方便管理
    SAVE_DIR = "Qwen_models/Qwen2.5-Coder-1.5B"
    
    download_hf_model(MODEL_REPO, SAVE_DIR)
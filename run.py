"""
BarbellGPT 启动脚本

快速启动力量举训练智能助手。
"""

import subprocess
import sys
from pathlib import Path

def main():
    """启动BarbellGPT"""
    print("🏋️ 启动 BarbellGPT...")
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    
    # 启动Streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(project_root / "main.py"),
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
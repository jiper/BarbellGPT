"""
BarbellGPT 主程序

力量举训练智能助手的主程序入口。
"""

import os
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """设置日志配置"""
    from config import LOG_LEVEL, LOG_FILE
    
    # 配置loguru
    logger.remove()  # 移除默认处理器
    
    # 添加控制台处理器
    logger.add(
        sys.stderr,
        level=LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件处理器
    logger.add(
        LOG_FILE,
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days"
    )

def check_dependencies():
    """检查依赖是否安装"""
    required_packages = [
        'streamlit', 'langchain', 'langchain_community', 
        'chromadb', 'sentence_transformers', 'loguru'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.error("请运行: pip install -r requirements.txt")
        return False
    
    logger.info("所有依赖包检查通过")
    return True

def check_config():
    """检查配置"""
    from config import DASHSCOPE_API_KEY
    
    if not DASHSCOPE_API_KEY:
        logger.warning("未设置DASHSCOPE_API_KEY，LLM功能将不可用")
        logger.info("请在环境变量中设置DASHSCOPE_API_KEY")
        return False
    
    logger.info("配置检查通过")
    return True

def create_sample_data():
    """创建示例数据"""
    from knowledge.document_loader import DocumentLoader
    from knowledge.text_processor import TextProcessor
    from agents.rag_agent import RAGAgent
    
    try:
        # 检查是否有文档
        loader = DocumentLoader()
        doc_info = loader.get_document_info()
        
        if doc_info['supported_files'] == 0:
            logger.info("未找到文档，创建示例知识库...")
            
            # 创建示例文档
            sample_docs = [
                "深蹲是力量举三大项之一，主要锻炼下肢力量。正确的深蹲姿势包括：双脚与肩同宽，脚尖略微外展，下蹲时膝盖不超过脚尖，保持背部挺直。",
                "硬拉是锻炼全身力量的重要动作。起始姿势：双脚与肩同宽，双手握杠铃，背部挺直，臀部下沉。拉起时保持背部挺直，用腿部力量启动。",
                "卧推主要锻炼胸部和上肢力量。躺在卧推凳上，双脚平放地面，双手握杠铃，下放时控制速度，推起时呼气。注意肩胛骨收紧。",
                "力量举训练的基本原则：渐进超负荷，充分休息，正确技术，合理营养。建议每周训练3-4次，每次1-2小时。",
                "训练安全注意事项：充分热身，使用正确的重量，保持正确姿势，不要急于增加重量。如有不适立即停止训练。"
            ]
            
            # 创建文档对象
            from langchain.schema import Document
            documents = []
            for i, text in enumerate(sample_docs):
                doc = Document(
                    page_content=text,
                    metadata={
                        'source': 'sample_data',
                        'file_name': f'sample_{i+1}.txt',
                        'file_type': '.txt'
                    }
                )
                documents.append(doc)
            
            # 处理文档
            processor = TextProcessor()
            processed_docs = processor.process_documents(documents)
            
            # 添加到知识库
            agent = RAGAgent()
            success = agent.add_documents(processed_docs)
            
            if success:
                logger.info("示例知识库创建成功")
            else:
                logger.error("示例知识库创建失败")
                
        else:
            logger.info(f"找到 {doc_info['supported_files']} 个文档")
            
    except Exception as e:
        logger.error(f"创建示例数据失败: {e}")

def run_streamlit_app():
    """运行Streamlit应用"""
    try:
        from ui.chat_interface import ChatInterface
        
        # 创建并运行聊天界面
        chat_interface = ChatInterface()
        chat_interface.run()
        
    except Exception as e:
        logger.error(f"运行Streamlit应用失败: {e}")
        raise

def main():
    """主函数"""
    print("🏋️ BarbellGPT - 力量举训练智能助手")
    print("=" * 50)
    
    # 设置日志
    setup_logging()
    logger.info("BarbellGPT 启动中...")
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 检查配置
    check_config()
    
    # 创建示例数据
    create_sample_data()
    
    # 运行应用
    logger.info("启动Streamlit应用...")
    run_streamlit_app()

if __name__ == "__main__":
    main() 
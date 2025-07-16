"""
BarbellGPT 命令行版本

独立于前端界面的命令行交互版本，用于测试核心功能。
"""

import os
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.rag_agent import RAGAgent
from agents.conversation_manager import ConversationManager
from knowledge.document_loader import DocumentLoader
from knowledge.text_processor import TextProcessor

class BarbellGPTCLI:
    """BarbellGPT命令行界面"""
    
    def __init__(self):
        """初始化CLI"""
        self.rag_agent = None
        self.conversation_manager = None
        self.session_id = "cli_session"
        
    def initialize(self):
        """初始化系统"""
        print("🏋️ BarbellGPT - 力量举训练智能助手 (CLI版本)")
        print("=" * 60)
        
        try:
            print("正在初始化系统...")
            
            # 初始化代理
            self.rag_agent = RAGAgent()
            self.conversation_manager = ConversationManager()
            
            # 初始化对话
            self.conversation_manager.start_conversation(self.session_id)
            
            print("✅ 系统初始化完成！")
            return True
            
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            logger.error(f"CLI初始化失败: {e}")
            return False
    
    def load_documents(self):
        """加载文档到知识库"""
        try:
            print("\n📚 正在检查知识库...")
            
            # 检查是否有文档
            loader = DocumentLoader()
            doc_info = loader.get_document_info()
            
            if doc_info['supported_files'] > 0:
                print(f"找到 {doc_info['supported_files']} 个文档")
                
                # 加载并处理文档
                documents = loader.load_all_documents()
                if documents:
                    processor = TextProcessor()
                    processed_docs = processor.process_documents(documents)
                    
                    # 添加到知识库
                    success = self.rag_agent.add_documents(processed_docs)
                    if success:
                        print(f"✅ 成功加载 {len(processed_docs)} 个文档到知识库")
                    else:
                        print("❌ 文档加载失败")
                else:
                    print("⚠️ 没有找到可加载的文档")
            else:
                print("⚠️ 没有找到文档，将使用示例数据")
                self._create_sample_data()
                
        except Exception as e:
            print(f"❌ 加载文档失败: {e}")
            logger.error(f"加载文档失败: {e}")
    
    def _create_sample_data(self):
        """创建示例数据"""
        try:
            print("创建示例知识库...")
            
            # 示例文档
            sample_docs = [
                "深蹲是力量举三大项之一，主要锻炼下肢力量。正确的深蹲姿势包括：双脚与肩同宽，脚尖略微外展，下蹲时膝盖不超过脚尖，保持背部挺直。",
                "硬拉是锻炼全身力量的重要动作。起始姿势：双脚与肩同宽，双手握杠铃，背部挺直，臀部下沉。拉起时保持背部挺直，用腿部力量启动。",
                "卧推主要锻炼胸部和上肢力量。躺在卧推凳上，双脚平放地面，双手握杠铃，下放时控制速度，推起时呼气。注意肩胛骨收紧。",
                "力量举训练的基本原则：渐进超负荷，充分休息，正确技术，合理营养。建议每周训练3-4次，每次1-2小时。",
                "训练安全注意事项：充分热身，使用正确的重量，保持正确姿势，不要急于增加重量。如有不适立即停止训练。"
            ]
            
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
            success = self.rag_agent.add_documents(processed_docs)
            if success:
                print("✅ 示例知识库创建成功")
            else:
                print("❌ 示例知识库创建失败")
                
        except Exception as e:
            print(f"❌ 创建示例数据失败: {e}")
            logger.error(f"创建示例数据失败: {e}")
    
    def show_help(self):
        """显示帮助信息"""
        print("\n📖 使用帮助:")
        print("  /help     - 显示此帮助信息")
        print("  /status   - 显示系统状态")
        print("  /clear    - 清空对话历史")
        print("  /quit     - 退出程序")
        print("  /load     - 重新加载文档")
        print("\n💬 直接输入问题开始对话！")
    
    def show_status(self):
        """显示系统状态"""
        try:
            print("\n📊 系统状态:")
            
            # 代理信息
            if self.rag_agent:
                agent_info = self.rag_agent.get_agent_info()
                
                # 向量存储信息
                vector_info = agent_info.get('vector_store_info', {})
                print(f"  知识库文档数: {vector_info.get('document_count', 0)}")
                
                # LLM信息
                llm_info = agent_info.get('llm_info', {})
                print(f"  LLM模型: {llm_info.get('model_name', 'Unknown')}")
                print(f"  LLM状态: {'✅ 已连接' if llm_info.get('is_initialized') else '❌ 未连接'}")
            
            # 对话信息
            if self.conversation_manager:
                conv_info = self.conversation_manager.get_conversation_summary(self.session_id)
                print(f"  对话消息数: {conv_info.get('total_messages', 0)}")
            
        except Exception as e:
            print(f"❌ 获取状态失败: {e}")
    
    def chat(self, user_input: str):
        """处理用户输入"""
        try:
            # 添加用户消息到对话管理器
            self.conversation_manager.add_message(self.session_id, user_input, is_user=True)
            
            # 获取对话历史
            conversation_history = self.conversation_manager.get_conversation_history(
                self.session_id, limit=10
            )
            
            # 生成AI回答
            print("\n🤖 AI助手: ", end="", flush=True)
            ai_response = self.rag_agent.chat(user_input, conversation_history)
            print(ai_response)
            
            # 添加AI消息到对话管理器
            self.conversation_manager.add_message(self.session_id, ai_response, is_user=False)
            
        except Exception as e:
            print(f"\n❌ 处理失败: {e}")
            logger.error(f"处理用户输入失败: {e}")
    
    def clear_conversation(self):
        """清空对话历史"""
        try:
            self.conversation_manager.clear_conversation(self.session_id)
            print("✅ 对话历史已清空")
        except Exception as e:
            print(f"❌ 清空对话失败: {e}")
    
    def run(self):
        """运行CLI"""
        # 初始化系统
        if not self.initialize():
            return
        
        # 加载文档
        self.load_documents()
        
        # 显示帮助
        self.show_help()
        
        # 主循环
        print("\n" + "=" * 60)
        while True:
            try:
                # 获取用户输入
                user_input = input("\n💬 你: ").strip()
                
                # 处理特殊命令
                if user_input.startswith('/'):
                    command = user_input.lower()
                    
                    if command == '/help':
                        self.show_help()
                    elif command == '/status':
                        self.show_status()
                    elif command == '/clear':
                        self.clear_conversation()
                    elif command == '/quit':
                        print("👋 再见！")
                        break
                    elif command == '/load':
                        self.load_documents()
                    else:
                        print("❓ 未知命令，输入 /help 查看帮助")
                    
                    continue
                
                # 处理普通对话
                if user_input:
                    self.chat(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 再见！")
                break
            except EOFError:
                print("\n\n👋 再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {e}")
                logger.error(f"CLI运行错误: {e}")

def main():
    """主函数"""
    cli = BarbellGPTCLI()
    cli.run()

if __name__ == "__main__":
    main() 
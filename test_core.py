"""
核心功能测试脚本

快速测试BarbellGPT的各个核心模块。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_knowledge_modules():
    """测试知识库模块"""
    print("🧪 测试知识库模块...")
    
    try:
        from knowledge.document_loader import DocumentLoader
        from knowledge.text_processor import TextProcessor
        from knowledge.vectorizer import Vectorizer
        
        # 测试文档加载器
        loader = DocumentLoader()
        doc_info = loader.get_document_info()
        print(f"  ✅ 文档加载器: 找到 {doc_info['supported_files']} 个文档")
        
        # 测试文本处理器
        processor = TextProcessor()
        print(f"  ✅ 文本处理器: 块大小 {processor.chunk_size}")
        
        # 测试向量化器
        vectorizer = Vectorizer()
        dim = vectorizer.get_embedding_dimension()
        print(f"  ✅ 向量化器: 维度 {dim}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 知识库模块测试失败: {e}")
        return False

def test_core_modules():
    """测试核心模块"""
    print("🧪 测试核心模块...")
    
    try:
        from core.vector_store import ChromaVectorStore
        from core.retriever import HybridRetriever
        from core.llm_manager import LLMManager
        
        # 测试向量存储
        vector_store = ChromaVectorStore()
        info = vector_store.get_collection_info()
        print(f"  ✅ 向量存储: {info.get('document_count', 0)} 个文档")
        
        # 测试检索器
        retriever = HybridRetriever(vector_store)
        print(f"  ✅ 检索器: 向量权重 {retriever.vector_weight}")
        
        # 测试LLM管理器
        llm_manager = LLMManager()
        llm_info = llm_manager.get_model_info()
        print(f"  ✅ LLM管理器: {llm_info.get('model_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 核心模块测试失败: {e}")
        return False

def test_agent_modules():
    """测试代理模块"""
    print("🧪 测试代理模块...")
    
    try:
        from agents.rag_agent import RAGAgent
        from agents.conversation_manager import ConversationManager
        
        # 测试对话管理器
        conv_manager = ConversationManager()
        conv_manager.start_conversation("test_session")
        print(f"  ✅ 对话管理器: 最大历史 {conv_manager.max_history}")
        
        # 测试RAG代理（不初始化LLM）
        print(f"  ✅ RAG代理: 模块加载成功")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 代理模块测试失败: {e}")
        return False

def test_simple_chat():
    """测试简单对话"""
    print("🧪 测试简单对话...")
    
    try:
        from agents.rag_agent import RAGAgent
        from agents.conversation_manager import ConversationManager
        
        # 初始化组件
        conv_manager = ConversationManager()
        conv_manager.start_conversation("test_session")
        
        # 测试对话
        user_message = "什么是深蹲？"
        conv_manager.add_message("test_session", user_message, is_user=True)
        
        history = conv_manager.get_conversation_history("test_session")
        print(f"  ✅ 对话测试: 历史长度 {len(history)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 对话测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🏋️ BarbellGPT - 核心功能测试")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("知识库模块", test_knowledge_modules),
        ("核心模块", test_core_modules),
        ("代理模块", test_agent_modules),
        ("简单对话", test_simple_chat),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        print()
    
    # 显示结果
    print("=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统可以正常运行。")
        print("\n💡 下一步:")
        print("  1. 运行 python cli.py 开始命令行交互")
        print("  2. 运行 streamlit run main.py 启动Web界面")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main() 
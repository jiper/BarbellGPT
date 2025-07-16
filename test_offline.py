"""
离线测试版本

不依赖网络下载的测试版本，用于验证基础功能。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_modules():
    """测试基础模块（不依赖网络）"""
    print("🧪 测试基础模块（离线模式）...")
    
    try:
        # 测试配置
        from config import BASE_DIR, DATA_DIR
        print(f"  ✅ 配置模块: 基础目录 {BASE_DIR}")
        
        # 测试文档加载器
        from knowledge.document_loader import DocumentLoader
        loader = DocumentLoader()
        doc_info = loader.get_document_info()
        print(f"  ✅ 文档加载器: 找到 {doc_info['supported_files']} 个文档")
        
        # 测试文本处理器
        from knowledge.text_processor import TextProcessor
        processor = TextProcessor()
        print(f"  ✅ 文本处理器: 块大小 {processor.chunk_size}")
        
        # 测试对话管理器
        from agents.conversation_manager import ConversationManager
        conv_manager = ConversationManager()
        conv_manager.start_conversation("test_session")
        print(f"  ✅ 对话管理器: 最大历史 {conv_manager.max_history}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 基础模块测试失败: {e}")
        return False

def test_vector_store():
    """测试向量存储（不依赖向量模型）"""
    print("🧪 测试向量存储...")
    
    try:
        from core.vector_store import ChromaVectorStore
        vector_store = ChromaVectorStore()
        info = vector_store.get_collection_info()
        print(f"  ✅ 向量存储: {info.get('document_count', 0)} 个文档")
        return True
        
    except Exception as e:
        print(f"  ❌ 向量存储测试失败: {e}")
        return False

def test_simple_conversation():
    """测试简单对话（不依赖LLM）"""
    print("🧪 测试简单对话...")
    
    try:
        from agents.conversation_manager import ConversationManager
        
        conv_manager = ConversationManager()
        conv_manager.start_conversation("test_session")
        
        # 添加测试消息
        conv_manager.add_message("test_session", "测试消息", is_user=True)
        conv_manager.add_message("test_session", "测试回复", is_user=False)
        
        history = conv_manager.get_conversation_history("test_session")
        print(f"  ✅ 对话测试: 历史长度 {len(history)}")
        
        # 获取摘要
        summary = conv_manager.get_conversation_summary("test_session")
        print(f"  ✅ 对话摘要: {summary.get('total_messages', 0)} 条消息")
        
        return True
        
    except Exception as e:
        print(f"  ❌ 对话测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🏋️ BarbellGPT - 离线功能测试")
    print("=" * 50)
    
    # 运行测试
    tests = [
        ("基础模块", test_basic_modules),
        ("向量存储", test_vector_store),
        ("简单对话", test_simple_conversation),
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
        print("🎉 离线测试通过！基础功能正常。")
        print("\n💡 下一步:")
        print("  1. 设置网络连接下载向量模型")
        print("  2. 设置 DASHSCOPE_API_KEY 启用LLM功能")
        print("  3. 运行 python cli.py 开始完整交互")
    else:
        print("⚠️ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main() 
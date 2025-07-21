import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.vector_store import ChromaVectorStore
from config import CHROMA_DB_PATH

def check_vector_store():
    vector_store = ChromaVectorStore()
    info = vector_store.get_collection_info()

    print(f"\n📁 检查路径: {CHROMA_DB_PATH}")
    print(f"📚 集合名: {info.get('collection_name', '未设置')}")
    print(f"📄 文档总数: {info.get('document_count', 0)}")

    if info.get("document_count", 0) > 0:
        print("✅ 向量数据库已存在且有效！")
    else:
        print("❌ 未检测到有效的向量数据")

if __name__ == "__main__":
    check_vector_store()

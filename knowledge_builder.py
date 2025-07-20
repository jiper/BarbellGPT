"""
知识库构建脚本

用于从文档构建向量数据库，支持增量更新。
"""

import os
from pathlib import Path
from loguru import logger
from knowledge.document_loader import DocumentLoader
from knowledge.text_processor import TextProcessor
from knowledge.vectorizer import Vectorizer
from core.vector_store import ChromaVectorStore
from config import CHROMA_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP

class KnowledgeBaseBuilder:
    """知识库构建器"""
    
    def __init__(self):
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        self.vectorizer = Vectorizer()
        self.vector_store = ChromaVectorStore()
        
    def build_knowledge_base(self, force_rebuild: bool = False):
        """
        构建知识库
        
        Args:
            force_rebuild: 是否强制重建（删除现有数据库）
        """
        logger.info("开始构建知识库...")
        
        # 1. 检查文档
        doc_info = self.document_loader.get_document_info()
        logger.info(f"文档统计: {doc_info}")
        
        if doc_info['supported_files'] == 0:
            logger.warning("没有找到支持的文档文件")
            return False
            
        # 2. 加载文档
        documents = self.document_loader.load_all_documents()
        logger.info(f"加载了 {len(documents)} 个文档")
        
        # 3. 文本处理
        processed_chunks = self.text_processor.process_documents(documents)
        logger.info(f"生成了 {len(processed_chunks)} 个文本块")
        
        # 4. 向量化文档
        encoded_docs = self.vectorizer.encode_documents(processed_chunks)
        logger.info(f"向量化完成: {len(encoded_docs)} 个文档")
        
        # 5. 存储到向量数据库
        if force_rebuild:
            # 删除现有集合
            self.vector_store.delete_collection()
            # 重新初始化
            self.vector_store = ChromaVectorStore()
            
        success = self.vector_store.add_documents(encoded_docs)
        
        if success:
            # 显示统计信息
            info = self.vector_store.get_collection_info()
            logger.info(f"知识库构建完成！统计信息: {info}")
        else:
            logger.error("知识库构建失败")
            
        return success

def main():
    """主函数"""
    builder = KnowledgeBaseBuilder()
    
    # 检查是否有新文档
    doc_info = builder.document_loader.get_document_info()
    if doc_info['supported_files'] == 0:
        print("请在 data/documents/ 目录中放入文档文件")
        print("支持格式: PDF, TXT, DOCX, MD")
        return
        
    # 构建知识库
    success = builder.build_knowledge_base()
    
    if success:
        print("✅ 知识库构建成功！")
        print(f"📊 处理了 {doc_info['supported_files']} 个文档文件")
    else:
        print("❌ 知识库构建失败")

if __name__ == "__main__":
    main() 
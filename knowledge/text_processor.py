"""
文本处理器

负责文档文本的清洗、分块和预处理，为向量化做准备。
"""

import re
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"导入langchain模块失败: {e}")
    raise

class TextProcessor:
    """文本处理器，负责文档清洗、分块和预处理"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        初始化文本处理器
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        )
        
        # 中文文本清洗规则
        self.cleanup_patterns = [
            (r'\s+', ' '),  # 多个空白字符替换为单个空格
            (r'[^\w\s\u4e00-\u9fff。，！？；：""''（）【】\-]', ''),  # 保留中文、英文、数字和基本标点
            (r'^\s+|\s+$', ''),  # 去除首尾空白
        ]
        
    def clean_text(self, text: str) -> str:
        """
        清洗文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
            
        cleaned_text = text
        
        # 应用清洗规则
        for pattern, replacement in self.cleanup_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
            
        # 去除过短的文本
        if len(cleaned_text.strip()) < 10:
            return ""
            
        return cleaned_text.strip()
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        将文档分割成小块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分割后的文档块列表
        """
        if not documents:
            return []
            
        try:
            # 使用LangChain的文本分割器
            split_docs = self.text_splitter.split_documents(documents)
            
            # 清洗每个文档块
            cleaned_docs = []
            for doc in split_docs:
                cleaned_content = self.clean_text(doc.page_content)
                if cleaned_content:  # 只保留非空内容
                    doc.page_content = cleaned_content
                    cleaned_docs.append(doc)
                    
            logger.info(f"文档分割完成: {len(documents)} -> {len(cleaned_docs)} 块")
            return cleaned_docs
            
        except Exception as e:
            logger.error(f"文档分割失败: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """
        完整的文档处理流程：清洗 -> 分割 -> 后处理
        
        Args:
            documents: 原始文档列表
            
        Returns:
            处理后的文档块列表
        """
        if not documents:
            return []
            
        logger.info(f"开始处理 {len(documents)} 个文档")
        
        # 1. 清洗原始文档
        cleaned_docs = []
        for doc in documents:
            cleaned_content = self.clean_text(doc.page_content)
            if cleaned_content:
                doc.page_content = cleaned_content
                cleaned_docs.append(doc)
                
        logger.info(f"文档清洗完成: {len(documents)} -> {len(cleaned_docs)}")
        
        # 2. 分割文档
        split_docs = self.split_documents(cleaned_docs)
        
        # 3. 后处理：添加处理元数据
        processed_docs = []
        for i, doc in enumerate(split_docs):
            # 添加处理相关的元数据
            doc.metadata.update({
                'chunk_id': i,
                'chunk_size': len(doc.page_content),
                'processed': True,
                'processor': 'TextProcessor'
            })
            processed_docs.append(doc)
            
        logger.info(f"文档处理完成，共生成 {len(processed_docs)} 个文本块")
        return processed_docs
    
    def get_processing_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """
        获取文档处理的统计信息
        
        Args:
            documents: 文档列表
            
        Returns:
            处理统计信息
        """
        if not documents:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'avg_chunk_size': 0,
                'total_characters': 0
            }
            
        total_chars = sum(len(doc.page_content) for doc in documents)
        avg_chunk_size = total_chars / len(documents) if documents else 0
        
        return {
            'total_documents': len(documents),
            'total_chunks': len(documents),
            'avg_chunk_size': round(avg_chunk_size, 2),
            'total_characters': total_chars,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap
        }
    
    def filter_documents(self, documents: List[Document], 
                        min_length: int = 50, 
                        max_length: int = 5000) -> List[Document]:
        """
        根据长度过滤文档块
        
        Args:
            documents: 文档列表
            min_length: 最小长度
            max_length: 最大长度
            
        Returns:
            过滤后的文档列表
        """
        filtered_docs = []
        
        for doc in documents:
            content_length = len(doc.page_content)
            if min_length <= content_length <= max_length:
                filtered_docs.append(doc)
                
        logger.info(f"文档过滤完成: {len(documents)} -> {len(filtered_docs)}")
        return filtered_docs 
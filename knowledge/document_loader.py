"""
文档加载器

支持多种格式的文档加载，包括PDF、TXT、DOCX等格式。
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader,
        UnstructuredMarkdownLoader,
    )
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"导入langchain模块失败: {e}")
    raise

class DocumentLoader:
    """文档加载器，支持多种格式的文档加载"""
    
    def __init__(self, documents_dir: Optional[Path] = None):
        """
        初始化文档加载器
        
        Args:
            documents_dir: 文档目录路径，默认为config中配置的路径
        """
        from config import DOCUMENTS_DIR
        self.documents_dir = documents_dir or DOCUMENTS_DIR
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.docx': Docx2txtLoader,
            '.md': UnstructuredMarkdownLoader,
        }
        
    def load_document(self, file_path: Path) -> List[Document]:
        """
        加载单个文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            文档列表
        """
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension not in self.supported_extensions:
                logger.warning(f"不支持的文件格式: {file_extension}")
                return []
                
            loader_class = self.supported_extensions[file_extension]
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # 添加文件元数据
            for doc in documents:
                doc.metadata.update({
                    'source': str(file_path),
                    'file_name': file_path.name,
                    'file_type': file_extension,
                })
                
            logger.info(f"成功加载文档: {file_path.name}, 页数: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []
    
    def load_all_documents(self) -> List[Document]:
        """
        加载文档目录中的所有支持格式的文档
        
        Returns:
            所有文档的列表
        """
        all_documents = []
        
        if not self.documents_dir.exists():
            logger.warning(f"文档目录不存在: {self.documents_dir}")
            return all_documents
            
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                documents = self.load_document(file_path)
                all_documents.extend(documents)
                
        logger.info(f"总共加载了 {len(all_documents)} 个文档")
        return all_documents
    
    def get_document_info(self) -> Dict[str, Any]:
        """
        获取文档目录的统计信息
        
        Returns:
            文档统计信息
        """
        info = {
            'total_files': 0,
            'supported_files': 0,
            'unsupported_files': 0,
            'file_types': {},
        }
        
        if not self.documents_dir.exists():
            return info
            
        for file_path in self.documents_dir.rglob("*"):
            if file_path.is_file():
                info['total_files'] += 1
                file_extension = file_path.suffix.lower()
                
                if file_extension in self.supported_extensions:
                    info['supported_files'] += 1
                    info['file_types'][file_extension] = info['file_types'].get(file_extension, 0) + 1
                else:
                    info['unsupported_files'] += 1
                    
        return info 
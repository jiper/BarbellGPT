"""
向量存储管理器

基于ChromaDB的向量数据库管理，支持文档存储、检索和索引管理。
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np
from loguru import logger

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"导入chromadb模块失败: {e}")
    raise

class ChromaVectorStore:
    """ChromaDB向量存储管理器"""
    
    def __init__(self, db_path: Optional[Path] = None, collection_name: str = "barbellgpt"):
        """
        初始化向量存储管理器
        
        Args:
            db_path: 数据库路径，默认为config中配置的路径
            collection_name: 集合名称
        """
        from config import CHROMA_DB_PATH
        
        self.db_path = db_path or CHROMA_DB_PATH
        self.collection_name = collection_name
        
        # 确保数据库目录存在
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self._init_client()
        
    def _init_client(self):
        """初始化ChromaDB客户端"""
        try:
            # 使用新的ChromaDB客户端初始化方式
            self.client = chromadb.PersistentClient(path=str(self.db_path))
            logger.info(f"ChromaDB客户端初始化成功: {self.db_path}")
            
            # 获取或创建集合
            self.collection = self._get_or_create_collection()
            
        except Exception as e:
            logger.error(f"ChromaDB客户端初始化失败: {e}")
            raise
    
    def _get_or_create_collection(self) -> Collection:
        """获取或创建集合"""
        try:
            # 尝试获取现有集合
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"使用现有集合: {self.collection_name}")
        except Exception:
            # 创建新集合
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "BarbellGPT知识库向量存储"}
            )
            logger.info(f"创建新集合: {self.collection_name}")
            
        return collection
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        添加文档到向量数据库
        
        Args:
            documents: 包含文档、向量和元数据的字典列表
            
        Returns:
            是否添加成功
        """
        if not documents:
            return True
            
        try:
            # 准备数据
            ids = []
            embeddings = []
            metadatas = []
            texts = []
            
            for i, doc_data in enumerate(documents):
                doc_id = f"doc_{i}_{hash(doc_data['text'])}"
                ids.append(doc_id)
                embeddings.append(doc_data['embedding'].tolist())
                metadatas.append(doc_data['metadata'])
                texts.append(doc_data['text'])
            
            # 批量添加到集合
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )
            
            logger.info(f"成功添加 {len(documents)} 个文档到向量数据库")
            return True
            
        except Exception as e:
            logger.error(f"添加文档到向量数据库失败: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, 
               n_results: int = 5, 
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            搜索结果列表
        """
        try:
            # 执行搜索
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=filter_metadata
            )
            
            # 格式化结果
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'text': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'id': results['ids'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    formatted_results.append(result)
            
            logger.info(f"向量搜索完成，返回 {len(formatted_results)} 个结果")
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            集合统计信息
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'db_path': str(self.db_path)
            }
        except Exception as e:
            logger.error(f"获取集合信息失败: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """
        删除集合
        
        Returns:
            是否删除成功
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"成功删除集合: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"删除集合失败: {e}")
            return False
    
    def update_document(self, doc_id: str, 
                       new_text: str, 
                       new_embedding: np.ndarray,
                       new_metadata: Dict[str, Any]) -> bool:
        """
        更新文档
        
        Args:
            doc_id: 文档ID
            new_text: 新文本
            new_embedding: 新向量
            new_metadata: 新元数据
            
        Returns:
            是否更新成功
        """
        try:
            self.collection.update(
                ids=[doc_id],
                embeddings=[new_embedding.tolist()],
                metadatas=[new_metadata],
                documents=[new_text]
            )
            logger.info(f"成功更新文档: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"更新文档失败: {e}")
            return False
    
    def delete_document(self, doc_id: str) -> bool:
        """
        删除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            是否删除成功
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"成功删除文档: {doc_id}")
            return True
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False 
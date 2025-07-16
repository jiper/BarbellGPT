"""
混合检索器

结合语义搜索和关键词搜索的混合检索系统。
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np
from loguru import logger

try:
    from langchain.schema import Document
    from langchain_community.retrievers import BM25Retriever
except ImportError as e:
    logger.error(f"导入langchain模块失败: {e}")
    raise

class HybridRetriever:
    """混合检索器，结合向量搜索和关键词搜索"""
    
    def __init__(self, vector_store, vector_weight: float = 0.7):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储管理器
            vector_weight: 向量搜索权重 (0-1)
        """
        self.vector_store = vector_store
        self.vector_weight = vector_weight
        self.keyword_weight = 1.0 - vector_weight
        
        # 关键词检索器（用于文本匹配）
        self.bm25_retriever = None
        
    def setup_bm25_retriever(self, documents: List[Document]):
        """
        设置BM25关键词检索器
        
        Args:
            documents: 文档列表
        """
        try:
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            logger.info("BM25检索器设置完成")
        except Exception as e:
            logger.error(f"设置BM25检索器失败: {e}")
    
    def search(self, query: str, 
               query_embedding: np.ndarray,
               n_results: int = 5,
               filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        混合搜索
        
        Args:
            query: 查询文本
            query_embedding: 查询向量
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            混合搜索结果
        """
        try:
            # 1. 向量搜索
            vector_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results * 2,  # 获取更多候选结果
                filter_metadata=filter_metadata
            )
            
            # 2. 关键词搜索（如果有BM25检索器）
            keyword_results = []
            if self.bm25_retriever:
                try:
                    keyword_docs = self.bm25_retriever.get_relevant_documents(query)
                    keyword_results = [
                        {
                            'text': doc.page_content,
                            'metadata': doc.metadata,
                            'id': doc.metadata.get('id', ''),
                            'score': 1.0  # BM25没有距离分数
                        }
                        for doc in keyword_docs[:n_results * 2]
                    ]
                except Exception as e:
                    logger.warning(f"关键词搜索失败: {e}")
            
            # 3. 结果融合
            combined_results = self._merge_results(
                vector_results, keyword_results, n_results
            )
            
            logger.info(f"混合搜索完成，返回 {len(combined_results)} 个结果")
            return combined_results
            
        except Exception as e:
            logger.error(f"混合搜索失败: {e}")
            return []
    
    def _merge_results(self, vector_results: List[Dict], 
                      keyword_results: List[Dict], 
                      n_results: int) -> List[Dict]:
        """
        融合向量搜索和关键词搜索结果
        
        Args:
            vector_results: 向量搜索结果
            keyword_results: 关键词搜索结果
            n_results: 最终结果数量
            
        Returns:
            融合后的结果
        """
        # 创建结果映射
        result_map = {}
        
        # 处理向量搜索结果
        for i, result in enumerate(vector_results):
            doc_id = result.get('id', f'vec_{i}')
            if doc_id not in result_map:
                result_map[doc_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'id': doc_id,
                    'vector_score': 1.0 - (result.get('distance', 0) or 0),
                    'keyword_score': 0.0,
                    'combined_score': 0.0
                }
        
        # 处理关键词搜索结果
        for i, result in enumerate(keyword_results):
            doc_id = result.get('id', f'kw_{i}')
            if doc_id in result_map:
                result_map[doc_id]['keyword_score'] = result.get('score', 0.0)
            else:
                result_map[doc_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'id': doc_id,
                    'vector_score': 0.0,
                    'keyword_score': result.get('score', 0.0),
                    'combined_score': 0.0
                }
        
        # 计算综合分数
        for result in result_map.values():
            result['combined_score'] = (
                self.vector_weight * result['vector_score'] +
                self.keyword_weight * result['keyword_score']
            )
        
        # 按综合分数排序并返回前N个结果
        sorted_results = sorted(
            result_map.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )
        
        return sorted_results[:n_results]
    
    def semantic_search(self, query_embedding: np.ndarray, 
                       n_results: int = 5,
                       filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        纯语义搜索
        
        Args:
            query_embedding: 查询向量
            n_results: 返回结果数量
            filter_metadata: 元数据过滤条件
            
        Returns:
            语义搜索结果
        """
        return self.vector_store.search(
            query_embedding=query_embedding,
            n_results=n_results,
            filter_metadata=filter_metadata
        )
    
    def keyword_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        纯关键词搜索
        
        Args:
            query: 查询文本
            n_results: 返回结果数量
            
        Returns:
            关键词搜索结果
        """
        if not self.bm25_retriever:
            logger.warning("BM25检索器未设置，无法进行关键词搜索")
            return []
        
        try:
            docs = self.bm25_retriever.get_relevant_documents(query)
            return [
                {
                    'text': doc.page_content,
                    'metadata': doc.metadata,
                    'id': doc.metadata.get('id', ''),
                    'score': 1.0
                }
                for doc in docs[:n_results]
            ]
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return [] 
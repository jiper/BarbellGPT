"""
向量化器

使用中文向量模型将文本转换为向量表示，支持批量处理和缓存。
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    from langchain.schema import Document
except ImportError as e:
    logger.error(f"导入sentence_transformers模块失败: {e}")
    raise

class Vectorizer:
    """向量化器，负责文本到向量的转换"""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """
        初始化向量化器
        
        Args:
            model_name: 向量模型名称，默认使用config中的配置
            device: 计算设备，默认自动选择
        """
        from config import VECTOR_MODEL_NAME, VECTOR_MODEL_PROVIDER
        
        self.model_name = model_name or VECTOR_MODEL_NAME
        self.provider = VECTOR_MODEL_PROVIDER
        
        # 自动选择设备
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
            
        # 初始化模型
        self._load_model()
        
    def _load_model(self):
        """加载向量模型"""
        try:
            if self.provider == "dashscope":
                # 使用阿里云百炼的向量模型
                from dashscope import TextEmbedding
                # 阿里云百炼TextEmbedding不需要在初始化时传入model参数
                self.model = TextEmbedding()
                logger.info(f"正在加载阿里云百炼向量模型: {self.model_name}")
            else:
                # 使用本地SentenceTransformer模型
                logger.info(f"正在加载本地向量模型: {self.model_name} (设备: {self.device})")
                self.model = SentenceTransformer(self.model_name, device=self.device)
            
            logger.info("向量模型加载完成")
        except Exception as e:
            logger.error(f"向量模型加载失败: {e}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        编码单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            文本向量
        """
        try:
            if not text or not isinstance(text, str):
                return np.zeros(self.get_embedding_dimension())
            
            if self.provider == "dashscope":
                # 使用阿里云百炼API
                response = self.model.call(text)
                if response.status_code == 200:
                    embedding = np.array(response.output.embeddings[0].embedding)
                    return embedding
                else:
                    logger.error(f"阿里云百炼API调用失败: {response.message}")
                    return np.zeros(self.get_embedding_dimension())
            else:
                # 使用本地模型编码
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding
            
        except Exception as e:
            logger.error(f"文本编码失败: {e}")
            return np.zeros(self.get_embedding_dimension())
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        批量编码文本列表
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            文本向量矩阵
        """
        if not texts:
            return np.array([])
            
        try:
            # 过滤空文本
            valid_texts = [text for text in texts if text and isinstance(text, str)]
            
            if not valid_texts:
                if self.provider == "dashscope":
                    return np.zeros((len(texts), self.get_embedding_dimension()))
                else:
                    return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
                
            if self.provider == "dashscope":
                # 阿里云百炼批量编码
                embeddings = []
                for text in valid_texts:
                    response = self.model.call(text)
                    if response.status_code == 200:
                        embedding = np.array(response.output.embeddings[0].embedding)
                        embeddings.append(embedding)
                    else:
                        logger.error(f"阿里云百炼API调用失败: {response.message}")
                        embeddings.append(np.zeros(self.get_embedding_dimension()))
                embeddings = np.array(embeddings)
            else:
                # 使用本地模型批量编码
                embeddings = self.model.encode(
                    valid_texts, 
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=True
                )
            
            # 如果有些文本被过滤了，需要填充零向量
            if len(valid_texts) < len(texts):
                full_embeddings = np.zeros((len(texts), embeddings.shape[1]))
                valid_idx = 0
                for i, text in enumerate(texts):
                    if text and isinstance(text, str):
                        full_embeddings[i] = embeddings[valid_idx]
                        valid_idx += 1
                embeddings = full_embeddings
                
            return embeddings
            
        except Exception as e:
            logger.error(f"批量文本编码失败: {e}")
            if self.provider == "dashscope":
                return np.zeros((len(texts), self.get_embedding_dimension()))
            else:
                return np.zeros((len(texts), self.model.get_sentence_embedding_dimension()))
    
    def encode_documents(self, documents: List[Document], 
                        batch_size: int = 32) -> List[Dict[str, Any]]:
        """
        编码文档列表，返回文档和向量的组合
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
            
        Returns:
            包含文档和向量的字典列表
        """
        if not documents:
            return []
            
        try:
            # 提取文本内容
            texts = [doc.page_content for doc in documents]
            
            # 批量编码
            embeddings = self.encode_texts(texts, batch_size)
            
            # 组合结果
            results = []
            for i, doc in enumerate(documents):
                result = {
                    'document': doc,
                    'embedding': embeddings[i],
                    'text': doc.page_content,
                    'metadata': doc.metadata.copy()
                }
                results.append(result)
                
            logger.info(f"文档编码完成: {len(documents)} 个文档")
            return results
            
        except Exception as e:
            logger.error(f"文档编码失败: {e}")
            return []
    
    def get_embedding_dimension(self) -> int:
        """
        获取向量维度
        
        Returns:
            向量维度
        """
        if self.provider == "dashscope":
            # 阿里云百炼text-embedding-v4的维度是1024
            return 1024
        else:
            return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 第一个向量
            embedding2: 第二个向量
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            # 归一化向量
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            # 计算余弦相似度
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"相似度计算失败: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.get_embedding_dimension(),
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown')
        } 
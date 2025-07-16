"""
核心模块

提供向量数据库集成、LLM管理和检索功能。
"""

from .vector_store import ChromaVectorStore
from .retriever import HybridRetriever
from .llm_manager import LLMManager

__all__ = ["ChromaVectorStore", "HybridRetriever", "LLMManager"]

# 版本信息
__version__ = "0.1.0" 
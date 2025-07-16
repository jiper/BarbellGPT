"""
知识库管理模块

提供文档加载、处理和向量化的功能，支持多种文档格式和中文向量模型。
"""

from .document_loader import DocumentLoader
from .text_processor import TextProcessor
from .vectorizer import Vectorizer

__all__ = ["DocumentLoader", "TextProcessor", "Vectorizer"]

# 版本信息
__version__ = "0.1.0" 
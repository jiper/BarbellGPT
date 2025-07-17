"""
配置管理模块

负责加载和管理应用的所有配置项，支持环境变量和配置文件。
"""

import os
from pathlib import Path
from typing import Optional

# 基础路径
BASE_DIR = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
DOCUMENTS_DIR = DATA_DIR / "documents"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LOGS_DIR = BASE_DIR / "logs"

# 确保目录存在
for dir_path in [DATA_DIR, DOCUMENTS_DIR, PROCESSED_DATA_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 阿里云百炼配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL")
DASHSCOPE_MODEL_NAME = os.getenv("DASHSCOPE_MODEL_NAME", "qwen-plus")

# LangSmith配置
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "barbellgpt")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")  # 保持字符串

# 向量数据库配置
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", "./data/chroma_db"))
VECTOR_MODEL_NAME = os.getenv("VECTOR_MODEL_NAME", "text-embedding-v4")
VECTOR_MODEL_PROVIDER = os.getenv("VECTOR_MODEL_PROVIDER", "dashscope")

# 应用配置
DEBUG = os.getenv("DEBUG", "True").lower() == "true"
APP_PORT = int(os.getenv("APP_PORT", "8501"))
APP_HOST = os.getenv("APP_HOST", "localhost")

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "./logs/barbellgpt.log")

# RAG配置
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K", "5"))

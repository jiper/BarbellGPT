"""
LLM管理器

管理大语言模型的连接、调用和响应处理。
"""

import os
from typing import List, Dict, Any, Optional, Union
from loguru import logger

try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage, AIMessage
    from langchain.prompts import PromptTemplate
except ImportError as e:
    logger.error(f"导入langchain模块失败: {e}")
    raise

class LLMManager:
    """LLM管理器，负责模型连接和调用"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化LLM管理器
        
        Args:
            model_name: 模型名称，默认使用config中的配置
        """
        from config import DASHSCOPE_API_KEY, DASHSCOPE_MODEL_NAME
        
        self.api_key = DASHSCOPE_API_KEY
        self.model_name = model_name or DASHSCOPE_MODEL_NAME
        
        if not self.api_key:
            logger.warning("未设置DASHSCOPE_API_KEY，LLM功能将不可用")
            self.llm = None
        else:
            self._init_llm()
    
    def _init_llm(self):
        """初始化LLM模型"""
        try:
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=0.7,
                max_tokens=2000
            )
            logger.info(f"LLM模型初始化成功: {self.model_name}")
        except Exception as e:
            logger.error(f"LLM模型初始化失败: {e}")
            self.llm = None
    
    def generate_response(self, prompt: str, 
                         context: Optional[str] = None,
                         system_message: Optional[str] = None) -> str:
        """
        生成响应
        
        Args:
            prompt: 用户提示
            context: 上下文信息
            system_message: 系统消息
            
        Returns:
            LLM响应
        """
        if not self.llm:
            return "LLM模型未初始化，无法生成响应"
        
        try:
            # 构建消息列表
            messages = []
            
            # 系统消息
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            # 上下文信息
            if context:
                messages.append(SystemMessage(content=f"上下文信息:\n{context}"))
            
            # 用户消息
            messages.append(HumanMessage(content=prompt))
            
            # 生成响应
            response = self.llm.invoke(messages)
            
            logger.info("LLM响应生成成功")
            return response.content
            
        except Exception as e:
            logger.error(f"LLM响应生成失败: {e}")
            return f"生成响应时发生错误: {str(e)}"
    
    def generate_with_template(self, template: str, 
                              variables: Dict[str, Any]) -> str:
        """
        使用模板生成响应
        
        Args:
            template: 提示模板
            variables: 模板变量
            
        Returns:
            LLM响应
        """
        if not self.llm:
            return "LLM模型未初始化，无法生成响应"
        
        try:
            # 创建提示模板
            prompt_template = PromptTemplate(
                template=template,
                input_variables=list(variables.keys())
            )
            
            # 格式化提示
            formatted_prompt = prompt_template.format(**variables)
            
            # 生成响应
            messages = [HumanMessage(content=formatted_prompt)]
            response = self.llm.invoke(messages)
            
            logger.info("模板化LLM响应生成成功")
            return response.content
            
        except Exception as e:
            logger.error(f"模板化LLM响应生成失败: {e}")
            return f"生成响应时发生错误: {str(e)}"
    
    def batch_generate(self, prompts: List[str]) -> List[str]:
        """
        批量生成响应
        
        Args:
            prompts: 提示列表
            
        Returns:
            响应列表
        """
        if not self.llm:
            return ["LLM模型未初始化"] * len(prompts)
        
        responses = []
        for i, prompt in enumerate(prompts):
            try:
                messages = [HumanMessage(content=prompt)]
                response = self.llm.invoke(messages)
                responses.append(response.content)
                logger.info(f"批量生成进度: {i+1}/{len(prompts)}")
            except Exception as e:
                logger.error(f"批量生成失败 (提示 {i+1}): {e}")
                responses.append(f"生成失败: {str(e)}")
        
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息
        """
        return {
            'model_name': self.model_name,
            'is_initialized': self.llm is not None,
            'api_key_set': bool(self.api_key)
        }
    
    def update_model_params(self, temperature: Optional[float] = None,
                           max_tokens: Optional[int] = None):
        """
        更新模型参数
        
        Args:
            temperature: 温度参数
            max_tokens: 最大token数
        """
        if not self.llm:
            logger.warning("LLM模型未初始化，无法更新参数")
            return
        
        try:
            if temperature is not None:
                self.llm.temperature = temperature
            if max_tokens is not None:
                self.llm.max_tokens = max_tokens
                
            logger.info("模型参数更新成功")
        except Exception as e:
            logger.error(f"更新模型参数失败: {e}") 
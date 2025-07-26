"""
RAG代理

基于LangGraph的检索增强生成代理，结合知识库和LLM提供智能回答。
"""

from typing import TypedDict, List, Dict, Any, Optional
import numpy as np
from loguru import logger

try:
    from langgraph.graph import StateGraph, END
    from langgraph.prebuilt import ToolNode
    from langchain.schema import Document
    from langchain_core.messages import HumanMessage, AIMessage
    from typing import List
    from langchain_core.messages import HumanMessage, ToolMessageChunk
except ImportError as e:
    logger.error(f"导入langgraph模块失败: {e}")
    raise

from core.vector_store import ChromaVectorStore
from core.retriever import HybridRetriever
from core.llm_manager import LLMManager
from knowledge.vectorizer import Vectorizer
from langgraph.prebuilt.tool_node import ToolNode


# 定义状态类型
class AgentState(TypedDict):
    messages: List
    context: str
    response: str

class RAGAgent:
    """RAG检索增强生成代理"""
    
    def __init__(self, 
                 vector_store: Optional[ChromaVectorStore] = None,
                 retriever: Optional[HybridRetriever] = None,
                 llm_manager: Optional[LLMManager] = None,
                 vectorizer: Optional[Vectorizer] = None):
        """
        初始化RAG代理
        
        Args:
            vector_store: 向量存储管理器
            retriever: 检索器
            llm_manager: LLM管理器
            vectorizer: 向量化器
        """
        # 初始化组件
        self.vector_store = vector_store or ChromaVectorStore()
        self.retriever = retriever or HybridRetriever(self.vector_store)
        self.llm_manager = llm_manager or LLMManager()
        self.vectorizer = vectorizer or Vectorizer()
        
        # 构建LangGraph工作流
        self.workflow = self._build_workflow()
        
        logger.info("RAG代理初始化完成")
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("retrieve", self._retrieve_node)

        def _wrapped_generate(state):
            print("[DEBUG] ✅ generate node 确实执行了")
            return self._generate_node_stream(state)

        workflow.add_node("generate", _wrapped_generate, is_streaming=True)

        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        workflow.set_entry_point("retrieve")

        return workflow.compile()

    
        
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """检索节点：从知识库检索相关信息"""
        try:
            # 获取最新的用户消息
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            
            # 向量化查询
            query_embedding = self.vectorizer.encode_text(user_message)
            
            # 检索相关文档
            search_results = self.retriever.search(
                query=user_message,
                query_embedding=query_embedding,
                n_results=3
            )
            
            # 构建上下文
            context = self._build_context(search_results)
            
            logger.info(f"检索完成，找到 {len(search_results)} 个相关文档")
            
            # 返回 TypedDict
            return {
                "messages": messages,
                "context": context,
                "response": ""
            }
            
        except Exception as e:
            logger.error(f"检索节点执行失败: {e}")
            return {
                "messages": state.get("messages", []),
                "context": "检索失败，无法获取相关信息。",
                "response": ""
            }
    
    
    def _generate_node(self, state: AgentState) -> AgentState:
        """生成节点：基于检索结果生成回答"""
        try:
            messages = state.get("messages", [])
            user_message = messages[-1].content if messages else ""
            
            # 构建系统提示
            system_prompt = self._build_system_prompt(state.get("context", ""))
            
            # 生成回答
            response = self.llm_manager.generate_response(
                prompt=user_message,
                context=state.get("context", ""),
                system_message=system_prompt
            )
            
            logger.info("回答生成完成")
            
            # 返回 TypedDict
            return {
                "messages": messages,
                "context": state.get("context", ""),
                "response": response
            }
            
        except Exception as e:
            logger.error(f"生成节点执行失败: {e}")
            return {
                "messages": state.get("messages", []),
                "context": state.get("context", ""),
                "response": f"生成回答时发生错误: {str(e)}"
            }

    from typing import Generator

    def _generate_node_stream(self, state: AgentState):
        messages = state.get("messages", [])
        user_message = messages[-1].content if messages else ""
        context = state.get("context", "")
        system_prompt = self._build_system_prompt(context)

        for chunk in self.llm_manager.stream_response(
            prompt=user_message,
            context=context,
            system_message=system_prompt
        ):
            yield {"response": chunk}  # 必须是 dict，不能是 ToolMessageChunk 等对象




    
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """构建上下文信息"""
        if not search_results:
            return "没有找到相关的知识库信息。"
        
        context_parts = ["基于以下知识库信息回答用户问题：\n"]
        
        for i, result in enumerate(search_results, 1):
            text = result.get('text', '')
            metadata = result.get('metadata', {})
            source = metadata.get('file_name', '未知来源')
            
            context_parts.append(f"信息{i} (来源: {source}):\n{text}\n")
        
        return "\n".join(context_parts)
    
    def _build_system_prompt(self, context: str) -> str:
        """构建系统提示"""
        return f"""你是一个专业的力量举训练智能助手。请基于以下知识库信息，为用户提供准确、专业的回答。

{context}

回答要求：
1. 基于知识库信息，提供准确的专业建议
2. 如果知识库信息不足，请明确说明
3. 使用中文回答，语言简洁明了
4. 针对力量举训练相关问题提供具体指导
5. 注意安全性和科学性

请开始回答用户的问题。"""
    
    def chat(self, user_message: str, conversation_history: List = None) -> str:
        """
        与代理进行对话
        
        Args:
            user_message: 用户消息
            conversation_history: 对话历史
            
        Returns:
            代理回答
        """
        try:
            # 准备消息历史
            messages = conversation_history or []
            messages.append(HumanMessage(content=user_message))

            
            # 执行工作流
            result = self.workflow.invoke({"messages": messages, "context": "", "response": ""})
            
            return result["response"]
            
        except Exception as e:
            logger.error(f"对话执行失败: {e}")
            return f"抱歉，处理您的请求时发生错误: {str(e)}"

    def chat_stream(self, user_message: str, conversation_history: List = None):
        try:
            messages = conversation_history or []
            messages.append(HumanMessage(content=user_message))
            stream = self.workflow.stream({
                "messages": messages,
                "context": "",
                "response": ""
            })

            for event in stream:
                if event.get("type") == "tool":
                    output = event.get("output", {})
                    if isinstance(output, dict) and "response" in output:
                        yield output["response"]

        except Exception as e:
            import traceback
            yield f"❌ 生成失败: {str(e)}"
            print(traceback.format_exc())


    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        添加文档到知识库
        
        Args:
            documents: 文档列表
            
        Returns:
            是否添加成功
        """
        try:
            # 向量化文档
            doc_data = self.vectorizer.encode_documents(documents)
            
            # 添加到向量存储
            success = self.vector_store.add_documents(doc_data)
            
            if success:
                # 更新BM25检索器
                self.retriever.setup_bm25_retriever(documents)
                logger.info(f"成功添加 {len(documents)} 个文档到知识库")
            
            return success
            
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取代理信息"""
        return {
            "agent_type": "RAG Agent",
            "vector_store_info": self.vector_store.get_collection_info(),
            "llm_info": self.llm_manager.get_model_info(),
            "vectorizer_info": self.vectorizer.get_model_info()
        } 
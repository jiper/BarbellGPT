"""
聊天界面

基于Streamlit的聊天界面，提供用户友好的交互体验。
"""

import streamlit as st
import uuid
from typing import Optional
from loguru import logger

from agents.rag_agent import RAGAgent
from agents.conversation_manager import ConversationManager

class ChatInterface:
    """Streamlit聊天界面"""
    
    def __init__(self):
        """初始化聊天界面"""
        self.rag_agent = None
        self.conversation_manager = None
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """初始化会话状态"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        if 'agent_initialized' not in st.session_state:
            st.session_state.agent_initialized = False
    
    def initialize_agent(self):
        """初始化RAG代理"""
        try:
            with st.spinner("正在初始化智能助手..."):
                self.rag_agent = RAGAgent()
                self.conversation_manager = ConversationManager()
                st.session_state.agent_initialized = True
                logger.info("RAG代理初始化完成")
        except Exception as e:
            st.error(f"初始化失败: {str(e)}")
            logger.error(f"代理初始化失败: {e}")
    
    def render_header(self):
        """渲染页面头部"""
        st.title("🏋️ BarbellGPT - 力量举训练智能助手")
        st.markdown("---")
        
        # 显示系统状态
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.session_state.agent_initialized:
                st.success("✅ 系统就绪")
            else:
                st.warning("⚠️ 系统初始化中")
        
        with col2:
            st.metric("对话数", len(st.session_state.messages) // 2)
        
        with col3:
            if st.button("🔄 重新初始化"):
                st.session_state.agent_initialized = False
                st.rerun()
    
    def render_sidebar(self):
        """渲染侧边栏"""
        with st.sidebar:
            st.header("📊 系统信息")
            
            if st.session_state.agent_initialized and self.rag_agent:
                agent_info = self.rag_agent.get_agent_info()
                
                st.subheader("🤖 代理信息")
                st.write(f"类型: {agent_info.get('agent_type', 'Unknown')}")
                
                # 向量存储信息
                vector_info = agent_info.get('vector_store_info', {})
                st.write(f"知识库文档: {vector_info.get('document_count', 0)}")
                
                # LLM信息
                llm_info = agent_info.get('llm_info', {})
                st.write(f"模型: {llm_info.get('model_name', 'Unknown')}")
                st.write(f"状态: {'✅ 已连接' if llm_info.get('is_initialized') else '❌ 未连接'}")
            
            st.markdown("---")
            
            # 对话管理
            st.subheader("💬 对话管理")
            if st.button("🗑️ 清空对话"):
                st.session_state.messages = []
                if self.conversation_manager:
                    self.conversation_manager.clear_conversation(st.session_state.session_id)
                st.rerun()
            
            if st.button("📝 新对话"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.rerun()
            
            st.markdown("---")
            
            # 帮助信息
            st.subheader("❓ 使用帮助")
            st.markdown("""
            **如何使用：**
            1. 直接输入你的问题
            2. 系统会从知识库检索相关信息
            3. 基于检索结果生成专业回答
            
            **支持的问题类型：**
            - 力量举训练技巧
            - 动作要领指导
            - 训练计划制定
            - 安全注意事项
            """)
    
    def render_chat_messages(self):
        """渲染聊天消息"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    def process_user_input(self, user_input: str):
        """处理用户输入"""
        try:
            # 添加用户消息到界面
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 添加用户消息到对话管理器
            if self.conversation_manager:
                self.conversation_manager.add_message(
                    st.session_state.session_id, 
                    user_input, 
                    is_user=True
                )
            
            # 获取对话历史
            conversation_history = []
            if self.conversation_manager:
                conversation_history = self.conversation_manager.get_conversation_history(
                    st.session_state.session_id, 
                    limit=10
                )
            
            # 生成AI回答
            with st.chat_message("assistant"):
                with st.spinner("🤔 正在思考..."):
                    ai_response = self.rag_agent.chat(user_input, conversation_history)
                    
                    # 添加AI消息到对话管理器
                    if self.conversation_manager:
                        self.conversation_manager.add_message(
                            st.session_state.session_id, 
                            ai_response, 
                            is_user=False
                        )
                    
                    # 显示AI回答
                    st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    
        except Exception as e:
            error_msg = f"处理请求时发生错误: {str(e)}"
            st.error(error_msg)
            logger.error(f"处理用户输入失败: {e}")
    
    def render_chat_input(self):
        """渲染聊天输入框"""
        if prompt := st.chat_input("请输入你的问题..."):
            if not st.session_state.agent_initialized:
                st.warning("系统正在初始化，请稍候...")
                return
            
            self.process_user_input(prompt)
    
    def render_footer(self):
        """渲染页面底部"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>🏋️ BarbellGPT - 专业的力量举训练智能助手</p>
            <p>基于LangGraph和RAG技术构建</p>
        </div>
        """, unsafe_allow_html=True)
    
    def run(self):
        """运行聊天界面"""
        # 设置页面配置
        st.set_page_config(
            page_title="BarbellGPT - 力量举训练助手",
            page_icon="🏋️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 初始化代理
        if not st.session_state.agent_initialized:
            self.initialize_agent()
        
        # 渲染界面
        self.render_header()
        
        # 创建两列布局
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # 主聊天区域
            st.subheader("💬 与AI助手对话")
            
            # 显示聊天消息
            self.render_chat_messages()
            
            # 聊天输入
            self.render_chat_input()
        
        with col2:
            # 侧边栏
            self.render_sidebar()
        
        # 页面底部
        self.render_footer() 
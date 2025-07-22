"""
聊天界面

基于Streamlit的聊天界面，提供用户友好的交互体验。
"""

import streamlit as st
import uuid
from loguru import logger

from agents.rag_agent import RAGAgent
from agents.conversation_manager import ConversationManager


# ------------------------
# 状态初始化
# ------------------------
def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_initialized" not in st.session_state:
        st.session_state.agent_initialized = False

    if "rag_agent" not in st.session_state or "conversation_manager" not in st.session_state:
        try:
            with st.spinner("正在初始化智能助手..."):
                st.session_state.rag_agent = RAGAgent()
                st.session_state.conversation_manager = ConversationManager()
                st.session_state.agent_initialized = True
                logger.info("RAG代理初始化完成")
        except Exception as e:
            st.error(f"初始化失败: {e}")
            logger.error(f"RAG 初始化失败: {e}")


# ------------------------
# 聊天输入处理
# ------------------------
def process_user_input(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})

    cm = st.session_state.conversation_manager
    cm.add_message(st.session_state.session_id, user_input, is_user=True)

    history = cm.get_conversation_history(st.session_state.session_id, limit=10)

    with st.chat_message("assistant"):
        with st.spinner("🤔 正在思考..."):
            try:
                response = st.session_state.rag_agent.chat(user_input, history)
                cm.add_message(st.session_state.session_id, response, is_user=False)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error("处理失败: " + str(e))
                logger.error(f"处理用户输入失败: {e}")


# ------------------------
# 聊天消息渲染
# ------------------------
def render_messages():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def render_chat_input():
    if prompt := st.chat_input("请输入你的问题..."):
        if not st.session_state.agent_initialized:
            st.warning("系统未就绪，请稍候...")
        else:
            process_user_input(prompt)


def render_main_panel():
    st.subheader("💬 与 BarbellGPT 对话")
    render_messages()
    render_chat_input()


# ------------------------
# 侧边栏
# ------------------------
def render_sidebar():
    with st.sidebar:
        st.header("📊 系统信息")

        if st.session_state.agent_initialized:
            rag_agent = st.session_state.rag_agent
            try:
                info = rag_agent.get_agent_info()
                st.subheader("🤖 代理信息")
                st.write(f"类型: {info.get('agent_type', '未知')}")
                st.write(f"文档数: {info.get('vector_store_info', {}).get('document_count', 0)}")
                st.write(f"模型: {info.get('llm_info', {}).get('model_name', '未知')}")
                st.write(f"状态: {'✅ 已连接' if info.get('llm_info', {}).get('is_initialized', False) else '❌ 未连接'}")
            except Exception as e:
                st.error("获取代理信息失败")
                logger.error(f"获取代理信息失败: {e}")

        st.subheader("💬 对话管理")
        if st.button("🗑 清空对话"):
            st.session_state.messages = []
            st.session_state.conversation_manager.clear_conversation(st.session_state.session_id)
            st.rerun()

        if st.button("🆕 新对话"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 重新初始化"):
            for key in ["rag_agent", "conversation_manager", "agent_initialized"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("---")
        st.subheader("❓ 使用帮助")
        st.markdown("""
        **如何使用：**
        1. 输入你的问题
        2. 系统从知识库中检索相关信息
        3. 基于检索结果生成专业回答

        **支持的问题类型：**
        - 力量举训练技巧
        - 动作要领指导
        - 训练计划制定
        - 安全注意事项
        """)


# ------------------------
# 主入口
# ------------------------
def main():
    st.set_page_config(
        page_title="BarbellGPT - 力量举训练助手",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_state()

    st.title("🏋️ BarbellGPT - 力量举训练智能助手")
    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        render_main_panel()
    with col2:
        render_sidebar()

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        🏋️ Powered by LangGraph + Streamlit
    </div>
    """, unsafe_allow_html=True)


# ------------------------
# 对外统一接口（类封装）
# ------------------------
class ChatInterface:
    """对外保留类接口，兼容原有启动方式"""
    def run(self):
        main()

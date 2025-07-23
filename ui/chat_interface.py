import streamlit as st
import uuid
from loguru import logger

from agents.rag_agent import RAGAgent
from agents.conversation_manager import ConversationManager


# ------------------------
# 状态初始化（含首次 rerun）
# ------------------------
def init_state():
    rerun_needed = False

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "您好，我是 BarbellGPT 💪 力量举训练助手，有什么可以帮您？"
        }]
        rerun_needed = True

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

    if rerun_needed:
        st.rerun()


# ------------------------
# 聊天输入处理（流式输出）
# ------------------------
def process_user_input_stream(user_input: str):
    st.session_state.messages.append({"role": "user", "content": user_input})
    cm = st.session_state.conversation_manager
    cm.add_message(st.session_state.session_id, user_input, is_user=True)

    history = cm.get_conversation_history(st.session_state.session_id, limit=10)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in st.session_state.rag_agent.chat_stream(user_input, history):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"❌ 处理失败: {e}"
            logger.error(full_response)
        finally:
            placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    cm.add_message(st.session_state.session_id, full_response, is_user=False)


# ------------------------
# 渲染聊天记录
# ------------------------
def render_messages():
    for msg in st.session_state.messages:
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and content:
            with st.chat_message(role):
                st.markdown(content)


# ------------------------
# 主面板（仅聊天记录）
# ------------------------
def render_main_panel():
    render_messages()


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
            st.session_state.messages = [{
                "role": "assistant",
                "content": "您好，我是 BarbellGPT 💪 力量举训练助手，有什么可以帮您？"
            }]
            st.session_state.conversation_manager.clear_conversation(st.session_state.session_id)
            st.rerun()

        if st.button("🆕 新对话"):
            st.session_id = str(uuid.uuid4())
            st.session_state.messages = [{
                "role": "assistant",
                "content": "您好，新会话已开启，请输入问题 💬"
            }]
            st.rerun()

        if st.button("🔄 重新初始化"):
            for key in ["rag_agent", "conversation_manager", "agent_initialized"]:
                st.session_state.pop(key, None)
            st.rerun()

        st.markdown("---")
        st.subheader("❓ 使用帮助")
        st.markdown("""
        **如何使用：**
        1. 在底部输入框输入你的问题
        2. 系统自动检索知识库并生成回复

        **支持内容：**
        - 力量举训练技巧
        - 动作规范与纠错
        - 周期计划设计
        - 恢复策略与疲劳管理
        """)


# ------------------------
# 页面主函数
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

    if prompt := st.chat_input("请输入你的问题..."):
        if st.session_state.agent_initialized:
            process_user_input_stream(prompt)
        else:
            st.warning("系统未就绪，请稍候...")


# ------------------------
# 类封装接口
# ------------------------
class ChatInterface:
    def run(self):
        main()

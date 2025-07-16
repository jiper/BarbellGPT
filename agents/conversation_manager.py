"""
对话管理器

管理多轮对话的状态、历史和上下文。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

try:
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
except ImportError as e:
    logger.error(f"导入langchain模块失败: {e}")
    raise

class ConversationManager:
    """对话管理器，处理多轮对话和状态管理"""
    
    def __init__(self, max_history: int = 10):
        """
        初始化对话管理器
        
        Args:
            max_history: 最大历史记录数
        """
        self.max_history = max_history
        self.conversations: Dict[str, List[BaseMessage]] = {}
        self.conversation_metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.info("对话管理器初始化完成")
    
    def start_conversation(self, session_id: str) -> bool:
        """
        开始新对话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否成功开始
        """
        try:
            if session_id not in self.conversations:
                self.conversations[session_id] = []
                self.conversation_metadata[session_id] = {
                    'start_time': datetime.now(),
                    'message_count': 0,
                    'last_activity': datetime.now()
                }
                logger.info(f"开始新对话: {session_id}")
                return True
            else:
                logger.warning(f"对话已存在: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"开始对话失败: {e}")
            return False
    
    def add_message(self, session_id: str, 
                   message: str, 
                   is_user: bool = True) -> bool:
        """
        添加消息到对话历史
        
        Args:
            session_id: 会话ID
            message: 消息内容
            is_user: 是否为用户消息
            
        Returns:
            是否添加成功
        """
        try:
            if session_id not in self.conversations:
                self.start_conversation(session_id)
            
            # 创建消息对象
            if is_user:
                msg = HumanMessage(content=message)
            else:
                msg = AIMessage(content=message)
            
            # 添加到历史
            self.conversations[session_id].append(msg)
            
            # 更新元数据
            self.conversation_metadata[session_id]['message_count'] += 1
            self.conversation_metadata[session_id]['last_activity'] = datetime.now()
            
            # 限制历史长度
            if len(self.conversations[session_id]) > self.max_history * 2:
                self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]
            
            logger.debug(f"添加消息到对话 {session_id}: {'用户' if is_user else 'AI'}")
            return True
            
        except Exception as e:
            logger.error(f"添加消息失败: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, 
                                limit: Optional[int] = None) -> List[BaseMessage]:
        """
        获取对话历史
        
        Args:
            session_id: 会话ID
            limit: 限制返回的消息数量
            
        Returns:
            对话历史列表
        """
        try:
            if session_id not in self.conversations:
                return []
            
            history = self.conversations[session_id]
            
            if limit:
                history = history[-limit:]
            
            return history
            
        except Exception as e:
            logger.error(f"获取对话历史失败: {e}")
            return []
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        获取对话摘要信息
        
        Args:
            session_id: 会话ID
            
        Returns:
            对话摘要
        """
        try:
            if session_id not in self.conversations:
                return {}
            
            metadata = self.conversation_metadata[session_id]
            history = self.conversations[session_id]
            
            # 统计消息类型
            user_messages = sum(1 for msg in history if isinstance(msg, HumanMessage))
            ai_messages = sum(1 for msg in history if isinstance(msg, AIMessage))
            
            return {
                'session_id': session_id,
                'start_time': metadata['start_time'],
                'last_activity': metadata['last_activity'],
                'total_messages': metadata['message_count'],
                'user_messages': user_messages,
                'ai_messages': ai_messages,
                'duration': (metadata['last_activity'] - metadata['start_time']).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"获取对话摘要失败: {e}")
            return {}
    
    def clear_conversation(self, session_id: str) -> bool:
        """
        清空对话历史
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否清空成功
        """
        try:
            if session_id in self.conversations:
                self.conversations[session_id] = []
                self.conversation_metadata[session_id] = {
                    'start_time': datetime.now(),
                    'message_count': 0,
                    'last_activity': datetime.now()
                }
                logger.info(f"清空对话历史: {session_id}")
                return True
            else:
                logger.warning(f"对话不存在: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"清空对话失败: {e}")
            return False
    
    def delete_conversation(self, session_id: str) -> bool:
        """
        删除对话
        
        Args:
            session_id: 会话ID
            
        Returns:
            是否删除成功
        """
        try:
            if session_id in self.conversations:
                del self.conversations[session_id]
                del self.conversation_metadata[session_id]
                logger.info(f"删除对话: {session_id}")
                return True
            else:
                logger.warning(f"对话不存在: {session_id}")
                return False
                
        except Exception as e:
            logger.error(f"删除对话失败: {e}")
            return False
    
    def get_all_conversations(self) -> List[Dict[str, Any]]:
        """
        获取所有对话的摘要信息
        
        Returns:
            所有对话摘要列表
        """
        try:
            summaries = []
            for session_id in self.conversations.keys():
                summary = self.get_conversation_summary(session_id)
                if summary:
                    summaries.append(summary)
            
            # 按最后活动时间排序
            summaries.sort(key=lambda x: x['last_activity'], reverse=True)
            
            return summaries
            
        except Exception as e:
            logger.error(f"获取所有对话失败: {e}")
            return []
    
    def cleanup_old_conversations(self, max_age_hours: int = 24) -> int:
        """
        清理旧对话
        
        Args:
            max_age_hours: 最大保留时间（小时）
            
        Returns:
            清理的对话数量
        """
        try:
            current_time = datetime.now()
            sessions_to_delete = []
            
            for session_id, metadata in self.conversation_metadata.items():
                age_hours = (current_time - metadata['last_activity']).total_seconds() / 3600
                if age_hours > max_age_hours:
                    sessions_to_delete.append(session_id)
            
            # 删除旧对话
            for session_id in sessions_to_delete:
                self.delete_conversation(session_id)
            
            logger.info(f"清理了 {len(sessions_to_delete)} 个旧对话")
            return len(sessions_to_delete)
            
        except Exception as e:
            logger.error(f"清理旧对话失败: {e}")
            return 0
    
    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        获取对话统计信息
        
        Returns:
            统计信息
        """
        try:
            total_conversations = len(self.conversations)
            total_messages = sum(
                metadata['message_count'] 
                for metadata in self.conversation_metadata.values()
            )
            
            return {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'max_history': self.max_history
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {} 
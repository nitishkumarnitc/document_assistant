import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from schemas import SessionState
from retrieval import SimulatedRetriever
from tools import get_all_tools, ToolLogger
from agent import create_workflow, AgentState
from prompts import MEMORY_SUMMARY_PROMPT


class DocumentAssistant:
    """
    The assistant creates and loads sessions and
    stores state/session data within a file.
    """

    def __init__(
            self,
            openai_api_key: str,
            model_name: str = "gpt-4o",
            temperature: float = 0.1,
            session_storage_path: str = "./sessions"
    ):
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model_name,
            temperature=temperature,
            base_url="https://openai.vocareum.com/v1"
        )

        # Initialize components
        self.retriever = SimulatedRetriever()
        self.tool_logger = ToolLogger(logs_dir="./logs")
        self.tools = get_all_tools(self.retriever, self.tool_logger)

        # Create workflow (compiled with checkpointer inside create_workflow)
        self.workflow = create_workflow(self.llm, self.tools)

        # Session management
        self.session_storage_path = session_storage_path
        os.makedirs(session_storage_path, exist_ok=True)

        # Current session
        self.current_session: Optional[SessionState] = None

    def start_session(self, user_id: str, session_id: Optional[str] = None) -> str:
        """Start a new session or resume an existing one."""
        if session_id and self._session_exists(session_id):
            # Load existing session
            self.current_session = self._load_session(session_id)
            print(f"Resumed session {session_id}")
        else:
            # Create new session
            session_id = session_id or str(uuid.uuid4())
            self.current_session = SessionState(
                session_id=session_id,
                user_id=user_id,
                conversation_history=[],
                document_context=[]
            )
            print(f"Started new session {session_id}")
        return session_id

    def _session_exists(self, session_id: str) -> bool:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        return os.path.exists(filepath)

    def _load_session(self, session_id: str) -> SessionState:
        filepath = os.path.join(self.session_storage_path, f"{session_id}.json")
        with open(filepath, 'r') as f:
            data = json.load(f)
        return SessionState(**data)

    def _save_session(self) -> None:
        if self.current_session:
            filepath = os.path.join(
                self.session_storage_path,
                f"{self.current_session.session_id}.json"
            )
            session_dict = self.current_session.dict()

            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj

            with open(filepath, 'w') as f:
                json.dump(session_dict, f, indent=2, default=serialize_datetime)

    def _get_conversation_summary(self, config) -> str:
        if not self.current_session or not self.current_session.conversation_history:
            return "No previous conversation."

        current_state = self.workflow.get_state(config).values

        summary = current_state.get("conversation_summary", [])
        return summary

    def _get_conversation_history(self, config) -> List[BaseMessage]:
        if not self.current_session or not self.current_session.conversation_history:
            return []

        current_state = self.workflow.get_state(config).values

        history = current_state.get("messages", [])
        return history


    def process_message(self, user_input: str) -> Dict[str, Any]:
        """Process a user message using the LangGraph workflow."""
        config = {
            "configurable": {
                "thread_id":self.current_session.session_id if self.current_session else None,
                "llm": self.llm,
                "tools": getattr(self, "tools", [])
            }
        }

        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")
        initial_state: AgentState = {
            "messages": [],
            "user_input": user_input,
            "intent": None,
            "next_step": "classify_intent",
            "conversation_history": self.current_session.conversation_history,
            "conversation_summary": self._get_conversation_summary(config),
            "active_documents": self.current_session.document_context,
            "current_response": None,
            "tools_used": [],
            "session_id": self.current_session.session_id,
            "user_id": self.current_session.user_id,
            # Initialise actions_taken list for this turn
            "actions_taken": []
        }
        try:
            # Invoke the workflow with a thread_id equal to the session_id
            final_state = self.workflow.invoke(initial_state, config=config)
            # Update session with new state
            if final_state.get("messages"):

                self.current_session.conversation_history.append({
                    "action": final_state.get("action",[]),
                    "intent": final_state.get("intent").dict() if final_state.get("intent") else None,
                    "summary": final_state.get("conversation_summary", "")
                })
                self.current_session.last_updated = datetime.now()
                if final_state.get("active_documents"):
                    self.current_session.document_context = list(set(
                        self.current_session.document_context +
                        final_state["active_documents"]
                    ))
                self._save_session()
            return {
                "success": True,
                "response": final_state.get("messages")[-1].content if final_state.get("messages") else None,
                "intent": final_state.get("intent").dict() if final_state.get("intent") else None,
                "tools_used": final_state.get("tools_used", []),
                "sources": final_state.get("active_documents", []),
                "actions_taken": final_state.get("actions_taken", []),
                "summary": final_state.get("conversation_summary", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
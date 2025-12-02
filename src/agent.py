from typing import TypedDict, Annotated, List, Dict, Any, Optional, Literal

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, tools_condition, ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
import re
import operator
from schemas import (
    UserIntent, SessionState,
    AnswerResponse, SummarizationResponse, CalculationResponse, UpdateMemoryResponse
)
from prompts import get_intent_classification_prompt, get_chat_prompt_template, MEMORY_SUMMARY_PROMPT

class AgentState(TypedDict):
    """
    The agent state object
    """
    # Current conversation
    user_input: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]

    # Intent and routing
    intent: Optional[UserIntent]
    next_step: str

    # Memory and context
    conversation_summary: str
    active_documents: Optional[List[str]]

    # Current task state
    current_response: Optional[Dict[str, Any]]
    tools_used: List[str]

    # Session management
    session_id: Optional[str]
    user_id: Optional[str]
    actions_taken: Annotated[List[str], operator.add]


def invoke_react_agent(response_schema: type[BaseModel], messages: List[BaseMessage], llm, tools) -> (
Dict[str, Any], List[str]):
    llm_with_tools = llm.bind_tools(
        tools
    )

    agent = create_react_agent(
        model=llm_with_tools,  # Use the bound model
        tools=tools,
        response_format=response_schema,
    )

    result = agent.invoke({"messages": messages})
    tools_used = [t.name for t in result.get("messages", []) if isinstance(t, ToolMessage)]

    return result, tools_used



def classify_intent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Classify user intent and update next_step. Also records that this
    function executed by appending "classify_intent" to actions_taken.
    """

    llm = config.get("configurable").get("llm")
    history = state.get("messages", []) or []
    user_input = state.get("user_input", "") or ""

    # Build a plain conversation history string the prompt expects
    conv_lines = []
    for m in history:
        content = getattr(m, "content", None)
        conv_lines.append(content if content is not None else str(m))
    conversation_history_text = "\n".join(conv_lines)

    # Configure LLM to produce structured UserIntent
    structured_llm = llm.with_structured_output(UserIntent)

    # Create and format the prompt
    prompt_template = get_intent_classification_prompt()
    prompt = prompt_template.format(user_input=user_input, conversation_history=conversation_history_text)

    # Invoke the LLM and assume we get a UserIntent instance back
    intent_obj = structured_llm.invoke(prompt)  # returns UserIntent

    # Extract normalized intent_type
    intent_type = getattr(intent_obj, "intent_type", "unknown")
    if isinstance(intent_type, str):
        intent_type = intent_type.strip().lower()
    else:
        intent_type = "unknown"

    # Map intent_type to next_step
    if intent_type == "qa":
        next_step = "qa_agent"
    elif intent_type == "summarization":
        next_step = "summarization_agent"
    elif intent_type == "calculation":
        next_step = "calculation_agent"
    else:
        next_step = "qa_agent"

    return {
        "actions_taken": ["classify_intent"],
        "intent": intent_obj,
        "next_step": next_step
    }


def qa_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle Q&A tasks and record the action.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    prompt_template = get_chat_prompt_template("qa")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(AnswerResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["qa_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def summarization_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle summarization tasks and record the action.
    Follows the same pattern as qa_agent but uses the summarization prompt
    and the SummarizationResponse structured schema.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")
    prompt_template = get_chat_prompt_template("summarization")
    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()
    result, tools_used = invoke_react_agent(SummarizationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["summarization_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def calculation_agent(state: AgentState, config: RunnableConfig) -> AgentState:
    """
    Handle calculation tasks and record the action.
    Follows the same pattern as qa_agent but uses the calculation prompt
    and the CalculationResponse structured schema.
    """
    llm = config.get("configurable").get("llm")
    tools = config.get("configurable").get("tools")

    # get the calculation chat prompt template and build messages
    prompt_template = get_chat_prompt_template("calculation")

    messages = prompt_template.invoke({
        "input": state["user_input"],
        "chat_history": state.get("messages", []),
    }).to_messages()

    result, tools_used = invoke_react_agent(CalculationResponse, messages, llm, tools)

    return {
        "messages": result.get("messages", []),
        "actions_taken": ["calculation_agent"],
        "current_response": result,
        "tools_used": tools_used,
        "next_step": "update_memory",
    }


def update_memory(state: AgentState, config:RunnableConfig) -> AgentState:
    """
    Update conversation memory and record the action.
    """
    llm = config.get("configurable").get("llm")

    prompt_with_history = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(MEMORY_SUMMARY_PROMPT),
        MessagesPlaceholder("chat_history"),
    ]).invoke({
        "chat_history": state.get("messages", []),
    })

    structured_llm = llm.with_structured_output(UpdateMemoryResponse)

    # LLM returns a UpdateMemoryResponse instance
    response: UpdateMemoryResponse = structured_llm.invoke(prompt_with_history)

    return {
        "actions_taken": ["update_memory"],
        "conversation_summary": response.conversation_summary,
        "active_documents": response.active_documents,
        "next_step": "end",
    }

def should_continue(state: AgentState) -> str:
    """Router function"""
    return state.get("next_step", "end")

def create_workflow(llm, tools):
    """
    Creates the LangGraph agents.
    Compiles the workflow with an InMemorySaver checkpointer to persist state.
    """
    # import here so callers don't need to add it to their imports at top of file
    from langgraph.checkpoint.memory import InMemorySaver

    workflow = StateGraph(AgentState)

    # Add nodes to the workflow
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("qa_agent", qa_agent)
    workflow.add_node("summarization_agent", summarization_agent)
    workflow.add_node("calculation_agent", calculation_agent)
    workflow.add_node("update_memory", update_memory)

    # Entry point
    workflow.set_entry_point("classify_intent")

    # Route classify_intent -> chosen agent based on the state's next_step value
    # should_continue(state) should return the next_step string (e.g. "qa_agent")
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {
            "qa_agent": "qa_agent",
            "summarization_agent": "summarization_agent",
            "calculation_agent": "calculation_agent",
            "end": END,
        }
    )

    # Connect each agent node to update_memory
    workflow.add_edge("qa_agent", "update_memory")
    workflow.add_edge("summarization_agent", "update_memory")
    workflow.add_edge("calculation_agent", "update_memory")

    # update_memory -> END
    workflow.add_edge("update_memory", END)

    # Compile with an in-memory checkpointer so thread state is persisted
    checkpointer = InMemorySaver()
    graph = workflow.compile(checkpointer=checkpointer)

    return graph

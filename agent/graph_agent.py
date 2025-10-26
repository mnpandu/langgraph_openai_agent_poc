from typing_extensions import TypedDict
from typing import Annotated, List, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# Import tools
from tools.data_tools import list_csv_datasets, get_dataset_details
from tools.code_tools import exec_python_code

# System Prompt
SYSTEM_PROMPT = (
    "You are a friendly, modern data analysis assistant. "
    "You can explore CSV datasets, describe their contents, "
    "and safely execute Python code for analysis or visualization."
)

# Build LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
tools = [list_csv_datasets, get_dataset_details, exec_python_code]
llm_with_tools = llm.bind_tools(tools)


# -------------------------------
# Graph Definition
# -------------------------------
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]


def analytics_agent(state: State):
    """Main LLM node"""
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


def run_tools(state: State):
    """Executes tools and returns ToolMessages"""
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, 'tool_calls', [])
    if not tool_calls:
        return {"messages": []}

    results = []
    for call in tool_calls:
        name, args, call_id = call["name"], call["args"], call["id"]
        tool_fn = next((f for f in tools if f.name == name), None)
        try:
            if not tool_fn:
                result = f"❌ Tool '{name}' not found."
            else:
                result = tool_fn.invoke(args)
        except Exception as e:
            result = f"❌ Error running {name}: {e}"
        results.append(ToolMessage(content=str(result), tool_call_id=call_id))
    return {"messages": results}


def route_to_tools(state: State):
    """Routes tool calls"""
    last = state["messages"][-1]
    return "run_tools" if getattr(last, 'tool_calls', []) else END


# -------------------------------
# Graph Compilation
# -------------------------------
def build_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("analytics_agent", analytics_agent)
    graph_builder.add_node("run_tools", run_tools)
    graph_builder.add_edge(START, "analytics_agent")
    graph_builder.add_edge("run_tools", "analytics_agent")
    graph_builder.add_conditional_edges("analytics_agent", route_to_tools)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from nodes.llm_nodes import llm_node
from nodes.tool_node import tool_node
from states import State
from tools import ROOT_DIR


def route_tools(state: State) -> str:
    """
    This is conditional_edge function to route to the ToolNode if the last message
    in the state has tool calls. Otherwise, route to the END node to complete the
    workflow.
    """
    messages = state.get("messages", [])
    if not messages:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    last_message = messages[-1]
    # If the last message is from the model and it contains a tool call request
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        return "tools"
    return END


def function_calling_agent(graph: CompiledStateGraph, user_input: str):
    user_message = HumanMessage(user_input)
    print(user_message.pretty_repr())
    input = {"messages": [user_message]}
    for event in graph.stream(input):
        for value in event.values():
            print(value["messages"][-1].pretty_repr())


graph_builder = StateGraph(State)
graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "llm")
graph_builder.add_conditional_edges(
    "llm",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {
        "tools": "tools",
        END: END,
    },
)

compiled_graph: CompiledStateGraph[State] = graph_builder.compile()

try:
    compiled_graph.get_graph().draw_mermaid_png(output_file_path=f"{ROOT_DIR}/images/fc_graph.png")
except Exception:
    # This requires some extra dependencies and is optional
    pass

if __name__ == "__main__":
    # user_input = "What is the weather in Miami?"
    # user_input = "What was the temperature variation in Miami today and last week?"
    user_input = "compare weather forecast for Toronto and Montréal"
    # user_input =  "What were the IBM stock prices on September 5, 2025?"

    # compiled_graph = create_react_agent(
    #     model=fc_llm,
    #     tools=tools,
    # )

    function_calling_agent(graph=compiled_graph, user_input=user_input)

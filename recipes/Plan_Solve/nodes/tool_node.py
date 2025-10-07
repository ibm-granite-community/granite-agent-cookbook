from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END
from langgraph.types import Command
from states import State
from tools.plan_complete import plan_complete
from tools.stock_market_tools import get_stock_price
from tools.weather_tools import (
    get_current_weather,
    get_geo_coordinates,
    get_weather_forecast,
    plot_weather_timeseries,
)
from typing_extensions import Dict, Literal


def get_tools_by_name() -> Dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    # Available tools for the agent
    tools = [
        get_stock_price,
        get_current_weather,
        get_weather_forecast,
        plot_weather_timeseries,
        get_geo_coordinates,
        plan_complete,
    ]
    return {tool.name: tool for tool in tools}


tools_by_name = get_tools_by_name()

# TODO use langgraph Prebuilt ToolNode
# from langgraph.prebuilt import ToolNode
# tool_node = ToolNode(tools=tools)


def tool_node(state: State) -> Command[Literal["llm", "__end__"]]:
    """Execute tool calls made by the LLM and return results.

    serves as the tool invocation node in the LangGraph workflow,
    executing tool calls defined by the LLM in previous steps.
    processes tool calls synchronously and returns the results as tool messages.

    Args:
        state (State): Current state containing messages with tool calls
                      to execute.

    Returns:
        Command[Literal["llm", "__end__"]]: Command object specifying the next node

    """
    print("***[TOOL_NODE] NODE***")
    messages = state.get("messages", [])
    most_recent_message = messages[-1]
    next_step = "llm"

    messages = []
    print(f"***[TOOL_NODE] TOOL CALLS {most_recent_message.tool_calls}***")

    for tool_call in most_recent_message.tool_calls:  # Handle synchronous tool calls
        tool_args = tool_call["args"]
        tool_name = tool_call["name"]

        tool_to_call = tools_by_name.get(tool_name)

        if tool_to_call:
            observation = tool_to_call.invoke(tool_args)
        else:
            observation = f"Tool {tool_name} not found"

        messages.append(ToolMessage(content=str(observation), name=tool_name, tool_call_id=tool_call["id"]))

        if tool_name == "plan_complete":
            next_step = END
        print(f"***[TOOL_NODE] RESULT {observation} {next_step}***")
    return Command(goto=next_step, update={"messages": messages})

from chains.function_calling_llm import fc_llm
from states import State
from tools.plan_complete import plan_complete
from tools.stock_market_tools import get_stock_price
from tools.weather_tools import (
    get_current_weather,
    get_geo_coordinates,
    get_weather_forecast,
    plot_weather_timeseries,
)

tools = [get_stock_price, get_current_weather, get_weather_forecast, plot_weather_timeseries, get_geo_coordinates, plan_complete]
llm_with_tools = fc_llm.bind_tools(tools)


def llm_node(state: State) -> State:
    messages = state.get("messages", [])
    response_message = llm_with_tools.invoke(messages)
    return State(messages=[response_message])

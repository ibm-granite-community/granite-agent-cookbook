from langgraph.graph import END, START, StateGraph
from nodes.execute_step_node import execute_step
from nodes.planner_nodes import planner_node, replanner_node, should_end
from states import PlanSolve
from tools import ROOT_DIR

graph = StateGraph(PlanSolve)
graph.add_node("planner_node", planner_node)
# from function_calling_agent import compiled_graph # TODO directly use subgraph as node
# graph.add_node("function_calling_agent",compiled_graph)
graph.add_node("function_calling_agent", execute_step)
graph.add_node("replan", replanner_node)

graph.add_edge(START, "planner_node")
graph.add_edge("planner_node", "function_calling_agent")
graph.add_edge("function_calling_agent", "replan")
graph.add_conditional_edges(
    "replan",
    should_end,
    {"function_calling_agent": "function_calling_agent", END: END},
)

planner_agent = graph.compile()


planner_agent.get_graph(xray=True).draw_mermaid_png(output_file_path=f"{ROOT_DIR}/images/plan_solve.png")


if __name__ == "__main__":
    # user_input = "What is the weather in Miami?"
    user_input = "compare weather forecast for Toronto and Montréal"
    # user_input =  "What were the IBM stock prices on September 5, 2025?"

    user_input = {"input": user_input}
    planner_agent.invoke(input=user_input)

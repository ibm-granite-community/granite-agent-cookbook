from typing import Dict

from function_calling_agent import compiled_graph as agent_executor
from langchain_core.messages import HumanMessage, filter_messages
from states import PlanSolve


def execute_step(state: PlanSolve) -> Dict:
    """Execute a step from the plan using a function-calling agent.

    This node is part of the Plan-Solve pattern where it takes the current plan
    and delegates execution to a specialized function-calling agent. The agent
    performs the planned actions using available tools and reports back on
    completion status.

    Args:
        state: Current PlanSolve state containing the plan to execute

    Returns:
        Updated state with past_steps (tool calls and responses) and
        plan_completed status indicating whether the plan has been fully executed
    """
    print("***[EXECUTE STEP] NODE***")
    plan = state["plan"]
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    task_formatted = f"""For the following plan: {plan_str}\n\nYou are tasked with executing these steps above."""
    messages = [HumanMessage(content=task_formatted)]
    agent_response = agent_executor.invoke({"messages": messages})
    plan_completed = False

    ai_messages = filter_messages(agent_response.get("messages"), include_types="ai")
    for ai_msg in ai_messages:
        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                if tool_call["name"] == "plan_complete":
                    plan_completed = True
                    print("***[EXECUTE STEP] plan_complete tool detected***")
                    break
    return {
        "past_steps": [
            (ai_m.tool_calls, tc_m.content)
            for ai_m, tc_m in zip(
                ai_messages,
                filter_messages(agent_response.get("messages"), include_types="tool"),
            )
        ],
        "plan_completed": plan_completed,
    }

from function_calling_agent import compiled_graph as agent_executor
from langchain_core.messages import HumanMessage, filter_messages
from states import PlanSolve


def execute_step(state: PlanSolve):
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

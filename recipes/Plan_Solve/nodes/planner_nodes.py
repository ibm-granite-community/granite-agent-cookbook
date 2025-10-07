from datetime import datetime

from chains.plannig_llm import planning_llm
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from prompts import planner_prompt, replanner_prompt
from states import Plan, PlanSolve


def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")


def planner_node(state: PlanSolve):
    """Generate an initial plan to solve the user's request.

    This node creates the first plan by analyzing the user's input and generating
    a structured sequence of steps to accomplish the task. It uses a structured
    LLM to ensure the plan follows a consistent format.

    Args:
        state: Current state containing the user's input request

    Returns:
        Dictionary containing the generated plan steps and completion status
    """
    print("***[PLANNER] NODE***")
    input = state["input"]
    system_message = planner_prompt.format(date=get_today_str())

    structured_planner_llm = planning_llm.with_structured_output(Plan)

    plan = structured_planner_llm.invoke([SystemMessage(content=system_message), HumanMessage(content=input)])
    return {"plan": plan.steps, "plan_completed": False}


def replanner_node(state: PlanSolve):
    """Update the plan based on previous execution results and progress.

    This node revises the existing plan by considering what steps have already
    been completed and their outcomes. It generates a new plan that accounts
    for the current state and any new information discovered during execution.

    Args:
        state: Current state containing the original plan and past step results

    Returns:
        Dictionary containing the updated plan steps
    """
    print("***[REPLANNER] NODE***")
    input = state.get("input")
    plan = state.get("plan")
    past_steps = state.get("past_steps")
    # format state variables for prompt
    plan_str = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(plan))
    past_steps_str = "\n".join(f"{i + 1}. {step[0]} and result was \n{step[1]}\n\n" for i, step in enumerate(past_steps))

    system_message = replanner_prompt.format(input=input, plan=plan_str, past_steps=past_steps_str)

    structured_replanner_llm = planning_llm.with_structured_output(Plan)

    replan = structured_replanner_llm.invoke([SystemMessage(content=system_message)])

    return {"plan": replan.steps}


def should_end(state: PlanSolve):
    """Conditional edge to Determine whether the planning-execution cycle should terminate.

    Checks if the plan has been completed or if there are no more steps
    to execute, in which case the workflow should end. Otherwise,
    continues to the function calling agent for step execution.

    Args:
        state: Current state containing plan and completion status

    Returns:
        Either END to terminate the workflow or "function_calling_agent" to continue
    """
    if state.get("plan_completed", False) or len(state["plan"]) == 0:
        return END
    else:
        return "function_calling_agent"

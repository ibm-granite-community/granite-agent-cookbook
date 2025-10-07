from langchain_core.tools import tool


@tool(parse_docstring=True)
def plan_complete() -> str:
    """
    Complete the plan executing with END node
    """
    print("***[PLAN_COMPLETE] TOOL CALLED***")
    return "Plan execution completed successfully"

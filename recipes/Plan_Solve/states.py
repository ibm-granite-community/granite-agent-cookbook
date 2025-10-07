import operator
from typing import List, Optional, Tuple

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


class Plan(BaseModel):
    """Plan to follow created by planner"""

    steps: List[str] = Field(description="plan steps to be executed in sorted order")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list[BaseMessage], add_messages]
    plan_completed: Optional[bool]


class PlanSolve(TypedDict):
    """
    State for the planner.
    Manages coordination between planner and executor for tracking progress and replan.
    """

    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    plan_completed: Optional[bool]

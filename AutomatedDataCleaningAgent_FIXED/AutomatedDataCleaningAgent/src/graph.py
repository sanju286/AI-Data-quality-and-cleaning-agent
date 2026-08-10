from typing import TypedDict, Any
import pandas as pd
from langgraph.graph import StateGraph, START, END

from .profiler import profile_dataframe
from .planner import llm_plan
from .cleaner import apply_cleaning_plan
from .validator import validate_dataframe


class AgentState(TypedDict, total=False):
    df: pd.DataFrame
    original: pd.DataFrame
    profile: Any
    plan: Any
    clean_log: list
    validation: dict
    attempts: int
    validation_history: list
    final_profile: Any
    explanation: str


def profile_node(state):
    return {
        "profile": profile_dataframe(state["df"])
    }


def plan_node(state):
    return {
        "plan": llm_plan(
            state["df"],
            state["profile"],
            state.get("validation_errors", []),
        )
    }


def clean_node(state):
    cleaned, log = apply_cleaning_plan(
        state["df"],
        state["plan"],
    )

    return {
        "df": cleaned,
        "clean_log": state.get("clean_log", []) + log,
        "attempts": state.get("attempts", 0) + 1,
    }


def validate_node(state):
    result = validate_dataframe(
        state["df"],
        state["profile"],
    )

    history = state.get("validation_history", [])
    history = history + [{
        "attempt": state.get("attempts", 1),
        "passed": result["passed"],
        "errors": result["errors"][:20],
    }]

    return {
        "validation": result,
        "validation_history": history,
        "validation_errors": result["errors"],
    }


def route_after_validation(state):
    if state["validation"]["passed"]:
        return END

    if state.get("attempts", 0) >= 3:
        return END

    return "profile"


builder = StateGraph(AgentState)

builder.add_node("profile", profile_node)
builder.add_node("plan", plan_node)
builder.add_node("clean", clean_node)
builder.add_node("validate", validate_node)

builder.add_edge(START, "profile")
builder.add_edge("profile", "plan")
builder.add_edge("plan", "clean")
builder.add_edge("clean", "validate")

builder.add_conditional_edges(
    "validate",
    route_after_validation,
    ["profile", END],
)

agent = builder.compile()


def run_agent(df: pd.DataFrame):
    result = agent.invoke({
        "df": df.copy(),
        "original": df.copy(),
        "attempts": 0,
        "validation_history": [],
        "validation_errors": [],
        "clean_log": [],
    })

    result["final_profile"] = profile_dataframe(result["df"])
    result["explanation"] = result["plan"].summary

    return result

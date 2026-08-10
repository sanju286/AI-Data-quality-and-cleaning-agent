import os
import pandas as pd
from .models import CleaningPlan, CleaningAction, Profile


def heuristic_plan(df: pd.DataFrame, profile: Profile) -> CleaningPlan:
    actions = []

    if profile.duplicate_rows:
        actions.append(
            CleaningAction(
                action="drop_duplicates",
                reason=f"Remove {profile.duplicate_rows} exact duplicate rows.",
            )
        )

    # One action per affected column. This avoids repeatedly applying the same
    # transformation when a column has multiple detected issues.
    missing_numeric = set()
    missing_categorical = set()
    outlier_columns = set()
    inconsistent_columns = set()

    for issue in profile.issues:
        if issue.issue_type == "missing_values" and issue.column:
            if pd.api.types.is_numeric_dtype(df[issue.column]):
                missing_numeric.add(issue.column)
            else:
                missing_categorical.add(issue.column)
        elif issue.issue_type == "numeric_outliers" and issue.column:
            outlier_columns.add(issue.column)
        elif issue.issue_type == "inconsistent_categorical" and issue.column:
            inconsistent_columns.add(issue.column)

    for col in sorted(missing_numeric):
        actions.append(
            CleaningAction(
                action="fill_missing_numeric",
                column=col,
                strategy="median",
                reason="Fill missing numeric values with the column median.",
            )
        )

    for col in sorted(missing_categorical):
        actions.append(
            CleaningAction(
                action="fill_missing_categorical",
                column=col,
                strategy="mode",
                reason="Fill missing categorical values with the most frequent category.",
            )
        )

    for col in sorted(inconsistent_columns):
        actions.append(
            CleaningAction(
                action="standardize_categories",
                column=col,
                reason="Normalize case and whitespace so equivalent categories share one canonical value.",
            )
        )

    for col in sorted(outlier_columns):
        bounds = profile.numeric_ranges.get(col, {})
        actions.append(
            CleaningAction(
                action="clip_numeric",
                column=col,
                lower=bounds.get("lower_bound"),
                upper=bounds.get("upper_bound"),
                reason="Clip extreme values using the same robust 3×IQR boundaries used by validation.",
            )
        )

    return CleaningPlan(
        actions=actions,
        summary=f"Generated {len(actions)} deterministic actions from the data-quality profile.",
    )


def llm_plan(df: pd.DataFrame, profile: Profile, validation_errors=None) -> CleaningPlan:
    """Use an LLM when configured, otherwise use the deterministic planner."""
    if not os.getenv("OPENAI_API_KEY"):
        plan = heuristic_plan(df, profile)

        # If validation failed, explicitly add corrective actions based on
        # the failing columns so the retry is genuinely feedback-driven.
        if validation_errors:
            existing = {(a.action, a.column) for a in plan.actions}
            for err in validation_errors:
                col = err.get("column")
                if not col or col not in df.columns:
                    continue

                key = ("clip_numeric", col)
                if pd.api.types.is_numeric_dtype(df[col]) and key not in existing:
                    bounds = profile.numeric_ranges.get(col, {})
                    plan.actions.append(
                        CleaningAction(
                            action="clip_numeric",
                            column=col,
                            lower=bounds.get("lower_bound"),
                            upper=bounds.get("upper_bound"),
                            reason="Corrective action generated from Pandera validation feedback.",
                        )
                    )
                    existing.add(key)

        plan.summary = (
            f"{plan.summary} "
            + ("Retry plan incorporates validation feedback." if validation_errors
               else "Planner mode: deterministic fallback.")
        )
        return plan

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    structured = llm.with_structured_output(CleaningPlan)

    feedback = validation_errors or []

    prompt = f"""
You are a senior data-quality engineer.

Create a conservative cleaning plan for a pandas dataset.

Allowed actions:
- drop_duplicates
- fill_missing_numeric
- fill_missing_categorical
- normalize_text
- coerce_numeric
- clip_numeric
- standardize_categories

Never invent a column.
Never delete non-duplicate rows.
Use the supplied numeric bounds when clipping.
If validation_errors are supplied, prioritize fixing those exact columns.

PROFILE:
{profile.model_dump_json(indent=2)}

VALIDATION ERRORS:
{feedback}

COLUMNS:
{list(df.columns)}

Return only a structured CleaningPlan with a concise reason for every action.
"""
    return structured.invoke(prompt)

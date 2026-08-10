import pandas as pd
from .models import CleaningPlan


def apply_cleaning_plan(df: pd.DataFrame, plan: CleaningPlan):
    out = df.copy()
    log = []

    for action in plan.actions:

        if action.action == "drop_duplicates":
            before = len(out)
            out = out.drop_duplicates().reset_index(drop=True)
            changed = before - len(out)
            log.append({
                "action": action.action,
                "column": "dataset",
                "changed": changed,
                "reason": action.reason,
            })
            continue

        col = action.column
        if not col or col not in out.columns:
            log.append({
                "action": action.action,
                "column": col or "unknown",
                "changed": 0,
                "reason": "Skipped: column not found.",
            })
            continue

        if action.action == "fill_missing_numeric":
            s = pd.to_numeric(out[col], errors="coerce")
            missing = int(s.isna().sum())
            fill = float(s.median()) if s.notna().any() else 0.0
            out[col] = s.fillna(fill).astype(float)
            log.append({
                "action": action.action,
                "column": col,
                "changed": missing,
                "reason": f"Filled missing values with median {fill:.4g}.",
            })

        elif action.action == "fill_missing_categorical":
            s = out[col].astype("string")
            missing = int(s.isna().sum())
            mode = s.mode(dropna=True)
            fill = str(mode.iloc[0]) if len(mode) else "unknown"
            out[col] = s.fillna(fill).astype(str)
            log.append({
                "action": action.action,
                "column": col,
                "changed": missing,
                "reason": f"Filled missing values with mode '{fill}'.",
            })

        elif action.action in {"normalize_text", "standardize_categories"}:
            before = out[col].copy()
            normalized = (
                out[col]
                .astype("string")
                .map(lambda x: " ".join(x.strip().lower().split()) if pd.notna(x) else x)
            )
            # Validation requires non-null categorical values, so missing
            # categories are filled with the mode after normalization.
            mode = normalized.mode(dropna=True)
            fill = str(mode.iloc[0]) if len(mode) else "unknown"
            normalized = normalized.fillna(fill).astype(str)
            changed = int((before.astype(str) != normalized.astype(str)).sum())
            out[col] = normalized

            log.append({
                "action": action.action,
                "column": col,
                "changed": changed,
                "reason": action.reason,
            })

        elif action.action == "coerce_numeric":
            before_missing = int(out[col].isna().sum())
            out[col] = pd.to_numeric(out[col], errors="coerce")
            after_missing = int(out[col].isna().sum())
            log.append({
                "action": action.action,
                "column": col,
                "changed": max(after_missing - before_missing, 0),
                "reason": "Coerced values to numeric; invalid values become missing.",
            })

        elif action.action == "clip_numeric":
            s = pd.to_numeric(out[col], errors="coerce")
            lo = action.lower
            hi = action.upper

            if lo is not None and hi is not None:
                changed = int(((s < lo) | (s > hi)).sum())
                out[col] = s.clip(lower=lo, upper=hi)
            else:
                changed = 0

            # If clipping still leaves missing numeric values, fill with median.
            if out[col].isna().any():
                median = float(out[col].median()) if out[col].notna().any() else 0.0
                out[col] = out[col].fillna(median)

            out[col] = out[col].astype(float)

            log.append({
                "action": action.action,
                "column": col,
                "changed": changed,
                "reason": f"Clipped using validation-aligned bounds [{lo}, {hi}].",
            })

    return out, log

import pandas as pd
from .models import Profile, Issue


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def profile_dataframe(df: pd.DataFrame) -> Profile:
    """Create a reusable data-quality profile.

    Numeric validation boundaries are deliberately based on the same robust
    3*IQR rule used by the cleaner/validator, so detection and validation
    cannot disagree about what an outlier is.
    """
    issues = []
    missing = df.isna().sum().astype(int).to_dict()
    numeric_ranges = {}
    categorical_values = {}

    for col, count in missing.items():
        if count > 0:
            issues.append(
                Issue(
                    issue_type="missing_values",
                    column=col,
                    count=int(count),
                    examples=[],
                    severity="high" if count / max(len(df), 1) > 0.20 else "medium",
                )
            )

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        issues.append(
            Issue(
                issue_type="duplicates",
                column=None,
                count=duplicate_count,
                examples=[],
                severity="medium",
            )
        )

    for col in df.columns:
        series = df[col]

        if _is_numeric(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()

            if len(numeric):
                q1 = float(numeric.quantile(0.25))
                q3 = float(numeric.quantile(0.75))
                iqr = q3 - q1

                if iqr > 0:
                    lower = q1 - 3.0 * iqr
                    upper = q3 + 3.0 * iqr
                else:
                    lower = float(numeric.min())
                    upper = float(numeric.max())

                bad = numeric[(numeric < lower) | (numeric > upper)]

                numeric_ranges[col] = {
                    "min": float(numeric.min()),
                    "max": float(numeric.max()),
                    "median": float(numeric.median()),
                    "q01": float(numeric.quantile(0.01)),
                    "q99": float(numeric.quantile(0.99)),
                    "q1": q1,
                    "q3": q3,
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                }

                if len(bad):
                    issues.append(
                        Issue(
                            issue_type="numeric_outliers",
                            column=col,
                            count=int(len(bad)),
                            examples=bad.head(5).tolist(),
                            severity="low",
                        )
                    )

        else:
            values = series.dropna().astype(str)
            categorical_values[col] = values.value_counts().head(30).index.tolist()

            groups = {}
            for raw in values.unique():
                canonical = " ".join(raw.strip().lower().split())
                groups.setdefault(canonical, set()).add(raw)

            for variants in groups.values():
                if len(variants) > 1:
                    issues.append(
                        Issue(
                            issue_type="inconsistent_categorical",
                            column=col,
                            count=len(variants),
                            examples=sorted(list(variants))[:10],
                            severity="medium",
                        )
                    )

    return Profile(
        rows=len(df),
        columns=len(df.columns),
        missing_by_column=missing,
        duplicate_rows=duplicate_count,
        dtypes={c: str(df[c].dtype) for c in df.columns},
        numeric_ranges=numeric_ranges,
        categorical_values=categorical_values,
        issues=issues,
    )

from .profiler import profile_dataframe


def quality_score(df):
    """Simple explainable 0-100 data-quality score."""
    profile = profile_dataframe(df)

    rows = max(len(df), 1)
    missing_ratio = sum(profile.missing_by_column.values()) / max(rows * max(profile.columns, 1), 1)
    duplicate_ratio = profile.duplicate_rows / rows

    outlier_count = sum(
        i.count for i in profile.issues if i.issue_type == "numeric_outliers"
    )
    outlier_ratio = outlier_count / rows

    categorical_count = sum(
        i.count for i in profile.issues
        if i.issue_type == "inconsistent_categorical"
    )
    categorical_penalty = categorical_count / max(rows, 1)

    penalty = (
        45 * missing_ratio
        + 25 * duplicate_ratio
        + 20 * min(outlier_ratio, 1)
        + 10 * min(categorical_penalty, 1)
    )

    score = max(0.0, min(100.0, 100.0 - penalty * 100))
    return round(score, 1), profile

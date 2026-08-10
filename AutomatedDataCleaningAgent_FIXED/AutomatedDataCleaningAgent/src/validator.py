import pandas as pd
import pandera.pandas as pa


def build_schema(df: pd.DataFrame, profile):
    """Build validation rules from the SAME profile used for cleaning."""
    columns = {}

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            bounds = profile.numeric_ranges.get(col, {})
            checks = []

            if bounds:
                lower = bounds.get("lower_bound")
                upper = bounds.get("upper_bound")

                if lower is not None:
                    checks.append(pa.Check.ge(float(lower)))

                if upper is not None:
                    checks.append(pa.Check.le(float(upper)))

            columns[col] = pa.Column(
                float,
                checks=checks,
                nullable=False,
                coerce=True,
            )

        else:
            columns[col] = pa.Column(
                str,
                nullable=False,
                coerce=True,
            )

    return pa.DataFrameSchema(columns, strict=True)


def validate_dataframe(df: pd.DataFrame, profile):
    schema = build_schema(df, profile)

    try:
        schema.validate(df, lazy=True)

        return {
            "passed": True,
            "errors": [],
            "schema": schema,
        }

    except pa.errors.SchemaErrors as exc:
        errors = []

        for _, row in exc.failure_cases.head(100).iterrows():
            errors.append({
                "column": str(row.get("column")),
                "check": str(row.get("check")),
                "failure": str(row.get("failure_case")),
                "index": str(row.get("index")),
            })

        return {
            "passed": False,
            "errors": errors,
            "schema": schema,
        }

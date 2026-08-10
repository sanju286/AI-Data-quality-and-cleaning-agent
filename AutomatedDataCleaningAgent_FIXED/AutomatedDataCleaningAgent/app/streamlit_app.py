import os
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.graph import run_agent
from src.quality import quality_score


st.set_page_config(
    page_title="AI Data Quality Agent",
    page_icon="🧹",
    layout="wide",
)

st.title("🧹 AI Data Quality & Cleaning Agent")
st.caption("Profile → Plan → Clean → Validate → Self-Correct")

uploaded = st.file_uploader(
    "Upload CSV or Excel dataset",
    type=["csv", "xlsx"],
)


def read_upload(file):
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def metric_delta(before, after):
    return after - before


if uploaded:
    df = read_upload(uploaded)

    before_score, before_profile = quality_score(df)

    st.subheader("Dataset Overview")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing Cells", int(df.isna().sum().sum()))
    c4.metric("Duplicates", int(df.duplicated().sum()))
    c5.metric("Quality Score", f"{before_score}/100")

    with st.expander("Preview uploaded data"):
        st.dataframe(df.head(100), use_container_width=True)

    if st.button("🚀 Run AI Data Quality Agent", type="primary"):

        with st.spinner("Agent is profiling, planning, cleaning and validating..."):
            result = run_agent(df)

        cleaned = result["df"]
        after_score, after_profile = quality_score(cleaned)

        st.success("Workflow completed.")

        st.subheader("📊 Data Quality Improvement")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Quality Score",
            f"{after_score}/100",
            delta=f"{after_score - before_score:+.1f}",
        )

        c2.metric(
            "Missing Cells",
            int(cleaned.isna().sum().sum()),
            delta=-int(df.isna().sum().sum()) + int(cleaned.isna().sum().sum()),
        )

        c3.metric(
            "Duplicates",
            int(cleaned.duplicated().sum()),
            delta=-int(df.duplicated().sum()) + int(cleaned.duplicated().sum()),
        )

        c4.metric(
            "Validation",
            "PASS" if result["validation"]["passed"] else "FAIL",
        )

        st.subheader("🤖 Agent Execution")

        mode = (
            "LLM Planner"
            if os.getenv("OPENAI_API_KEY")
            else "Deterministic Planner"
        )

        a1, a2, a3 = st.columns(3)
        a1.metric("Planner", mode)
        a2.metric("Agent Attempts", result.get("attempts", 1))
        a3.metric(
            "Issues Remaining",
            len(after_profile.issues),
        )

        st.subheader("🔎 Detected Issues")

        issue_rows = [issue.model_dump() for issue in before_profile.issues]

        if issue_rows:
            st.dataframe(
                pd.DataFrame(issue_rows),
                use_container_width=True,
            )
        else:
            st.success("No quality issues detected.")

        st.subheader("🧠 Cleaning Plan")

        st.write(result["plan"].summary)

        for action in result["plan"].actions:
            st.markdown(
                f"- **{action.action}** → `{action.column or 'dataset'}`  \n"
                f"  {action.reason}"
            )

        st.subheader("🛠 Cleaning Audit Log")

        if result["clean_log"]:
            st.dataframe(
                pd.DataFrame(result["clean_log"]),
                use_container_width=True,
            )

        st.subheader("🔄 Validation & Self-Correction")

        history = result.get("validation_history", [])

        if history:
            for item in history:
                status = "✅ PASS" if item["passed"] else "❌ FAIL"
                st.markdown(
                    f"**Attempt {item['attempt']} — {status}**"
                )

                if item["errors"]:
                    st.dataframe(
                        pd.DataFrame(item["errors"]),
                        use_container_width=True,
                    )

        if result["validation"]["passed"]:
            st.success(
                "Pandera validation passed. The cleaned dataset satisfies "
                "the validation rules generated from the data profile."
            )
        else:
            st.warning(
                "Validation still has unresolved issues after the retry limit."
            )

        st.subheader("📈 Before vs After")

        comparison = pd.DataFrame({
            "Metric": [
                "Missing Cells",
                "Duplicate Rows",
                "Detected Issues",
                "Quality Score",
            ],
            "Before": [
                int(df.isna().sum().sum()),
                int(df.duplicated().sum()),
                len(before_profile.issues),
                before_score,
            ],
            "After": [
                int(cleaned.isna().sum().sum()),
                int(cleaned.duplicated().sum()),
                len(after_profile.issues),
                after_score,
            ],
        })

        st.dataframe(comparison, use_container_width=True)

        st.subheader("🧹 Cleaned Dataset")

        left, right = st.columns(2)

        with left:
            st.caption("Before")
            st.dataframe(df.head(50), use_container_width=True)

        with right:
            st.caption("After")
            st.dataframe(cleaned.head(50), use_container_width=True)

        csv_bytes = cleaned.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Cleaned CSV",
            data=csv_bytes,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )

else:
    st.info(
        "Upload a CSV or Excel dataset to start the agent. "
        "Without an API key, the project uses a deterministic planner; "
        "with OPENAI_API_KEY, the planner can use structured LLM output."
    )

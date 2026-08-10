from src.evaluation import make_clean_dataset, corrupt_dataset
from src.profiler import profile_dataframe
from src.graph import run_agent


def test_profiler_detects_core_issues():
    clean = make_clean_dataset(100)
    bad = corrupt_dataset(clean)
    profile = profile_dataframe(bad)

    types = {issue.issue_type for issue in profile.issues}

    assert "missing_values" in types
    assert "duplicates" in types
    assert "inconsistent_categorical" in types
    assert "numeric_outliers" in types


def test_agent_reduces_missing_and_duplicates():
    clean = make_clean_dataset(100)
    bad = corrupt_dataset(clean)

    result = run_agent(bad)
    cleaned = result["df"]

    assert cleaned.duplicated().sum() == 0
    assert cleaned.isna().sum().sum() == 0


def test_validation_history_exists():
    clean = make_clean_dataset(100)
    bad = corrupt_dataset(clean)

    result = run_agent(bad)

    assert len(result["validation_history"]) >= 1
    assert result["attempts"] <= 3

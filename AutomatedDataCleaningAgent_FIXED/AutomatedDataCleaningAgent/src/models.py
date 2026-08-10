from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "drop_duplicates",
    "fill_missing_numeric",
    "fill_missing_categorical",
    "normalize_text",
    "coerce_numeric",
    "clip_numeric",
    "standardize_categories",
]


class CleaningAction(BaseModel):
    action: ActionType
    column: Optional[str] = None
    strategy: Optional[str] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    mapping: Dict[str, str] = Field(default_factory=dict)
    reason: str = ""


class CleaningPlan(BaseModel):
    actions: List[CleaningAction] = Field(default_factory=list)
    summary: str = ""


class Issue(BaseModel):
    issue_type: str
    column: Optional[str] = None
    count: int = 0
    examples: List[Any] = Field(default_factory=list)
    severity: str = "medium"


class Profile(BaseModel):
    rows: int
    columns: int
    missing_by_column: Dict[str, int]
    duplicate_rows: int
    dtypes: Dict[str, str]
    numeric_ranges: Dict[str, Dict[str, float]]
    categorical_values: Dict[str, List[str]]
    issues: List[Issue]
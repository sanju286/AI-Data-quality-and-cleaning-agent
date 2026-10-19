Overview
An agentic data-quality system that automatically profiles tabular datasets, detects quality problems, creates a structured cleaning plan, applies safe Pandas transformations, validates the result with Pandera, and uses validation feedback to perform corrective retries.

Architecture
CSV / Excel
    |
    v
+----------------------+
| Data Profiler        |
| Pandas               |
+----------+-----------+
           |
           v
+----------------------+
| Cleaning Planner     |
| LLM / deterministic  |
+----------+-----------+
           |
           v
+----------------------+
| Cleaning Executor    |
| Pandas               |
+----------+-----------+
           |
           v
+----------------------+
| Pandera Validator    |
+----------+-----------+
           |
      PASS | FAIL
           | 
       +---+---+
       |       |
       v       v
     FINAL   Validation
             Feedback
                |
                v
             Re-profile
                |
                +------> Planner
Key improvements
1. Validation-aligned rules
The profiler calculates robust numeric boundaries using the 3×IQR rule. The validator uses the same boundaries. This prevents the earlier situation where the profiler and validator disagreed about whether a value was an outlier.

2. Feedback-driven self-correction
When Pandera fails, its column/check/failure information is placed back into the agent state. The next planning cycle receives that feedback and generates a corrective action.

3. Explainable quality score
The dashboard calculates an explainable 0–100 score using missing-value, duplicate, outlier and categorical-consistency penalties.

4. LLM + deterministic fallback
With OPENAI_API_KEY, the planner can use a structured LLM response. Without a key, the same workflow works with a deterministic rule-based planner.

Run
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# optional
copy .env.example .env

streamlit run app/streamlit_app.py
Test
pytest -q

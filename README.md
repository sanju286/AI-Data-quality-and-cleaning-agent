1. Open the project folder

Open the folder containing:

requirements.txt
app/
src/
tests/

Open Command Prompt in this folder.

2. Create a virtual environment
python -m venv .venv
3. Activate the virtual environment

Windows:

.venv\Scripts\activate

You should see:

(.venv)
4. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
5. Optional: Configure the LLM

Create a .env file in the project root:

OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

If no API key is provided, the application uses the deterministic fallback planner.

6. Run the Streamlit application
python -m streamlit run app\streamlit_app.py

The application will open in your browser.

7. Run tests
pytest -q
Quick Start
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app\streamlit_app.py

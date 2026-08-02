"""Pytest configuration shared by every test module.

Everything under app/ (app.py, ingest.py, and the config/ui/prompts/rag/
llm modules app.py is composed from) isn't packaged — no app/__init__.py,
no pyproject.toml entry point — it's run directly, the same way the app
itself is launched (`streamlit run app/app.py`, `python app/ingest.py`).
To import any of these as plain sibling modules from tests/ (e.g.
`import prompts`, `import rag`) without turning app/ into an installable
package, we add app/ to sys.path here, once, before any test module runs.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

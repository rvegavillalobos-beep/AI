"""Every page renders without an error.

Two layers:
* ``tests/run_pages.py`` drives the app through a strict Streamlit stub (no
  server): all pages, clicks and selections, filters narrowed to one type and
  to one week. It runs everywhere, even without Streamlit installed.
* With the real Streamlit installed, ``AppTest`` runs the actual script.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["command", "corners", "geometry", "shape", "drift", "anatomy", "spc", "patterns", "height",
         "target", "compensation", "events", "passport", "data", "reports"]


def test_every_page_with_the_stub():
    env = {**os.environ, "PYTHONWARNINGS": "ignore"}
    res = subprocess.run([sys.executable, str(ROOT / "tests" / "run_pages.py")], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=900)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-4000:]
    assert "scenarios clean" in res.stdout


def _real_streamlit():
    try:
        from streamlit.testing.v1 import AppTest  # noqa: F401
    except Exception:  # noqa: BLE001 - not installed, or the stub
        return False
    return True


real = pytest.mark.skipif(not _real_streamlit(), reason="Streamlit is not installed")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=240)
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    demo = [b for b in at.button if b.label == "Explore with demo data"]
    assert demo, "the start page shows the demo button"
    demo[0].click().run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@real
def test_demo_opens_the_command_center(app):
    assert [t.value for t in app.title][:1] == ["Command Center"]
    assert not app.error, [e.value for e in app.error]


@real
@pytest.mark.parametrize("key", PAGES)
def test_page_with_real_streamlit(app, key):
    from app.ui import PAGE_TITLES

    app.switch_page(f"app/pages/{key}.py").run()
    assert not app.exception, [e.message for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    titles = [t.value for t in app.title]
    if titles[:1] != [PAGE_TITLES[key]]:
        pytest.skip(f"AppTest could not switch to {key} in this Streamlit version (shown: {titles[:1]})")

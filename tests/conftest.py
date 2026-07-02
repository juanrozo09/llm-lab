# tests/conftest.py
import pytest


@pytest.fixture(autouse=True)
def _wide_terminal(monkeypatch):
    # Rich's Console falls back to an 80-column default when stdout isn't a
    # real tty (as under CliRunner), which wraps table cells far more
    # aggressively than in an actual terminal. Widen it so table-rendering
    # tests reflect realistic terminal widths instead of that fallback.
    monkeypatch.setenv("COLUMNS", "200")

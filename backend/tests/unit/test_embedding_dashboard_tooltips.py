import importlib.util
from pathlib import Path


def _load_toggle_help_from_scripts():
    """Load the toggle_help function from scripts/embedding_dashboard.py.

    This avoids relying on the scripts package being importable and
    keeps the test focused on the pure state transition helper.
    """

    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / "embedding_dashboard.py"
    spec = importlib.util.spec_from_file_location("embedding_dashboard", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return getattr(module, "toggle_help")


toggle_help = _load_toggle_help_from_scripts()


def test_toggle_help_initial_to_section():
    # When no section is active, clicking one should activate it.
    assert toggle_help("", "norms") == "norms"
    assert toggle_help("", "sampling") == "sampling"


def test_toggle_help_same_section_toggles_off():
    # Clicking the same active section again should clear it.
    assert toggle_help("norms", "norms") == ""
    assert toggle_help("sampling", "sampling") == ""


def test_toggle_help_switches_between_sections():
    # Clicking a different section should switch focus.
    assert toggle_help("norms", "dims") == "dims"
    assert toggle_help("dims", "pca") == "pca"


def test_toggle_help_idempotence_on_empty():
    # Clicking an empty section name should leave it unchanged.
    assert toggle_help("", "") == ""
    # If somehow active is an unknown value, clicking another still switches.
    assert toggle_help("unknown", "norms") == "norms"

"""Repository paths shared by the UI and command-line tools."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
NASDAQ_ROOT = DATA_ROOT / "nasdaq_10y"
GROUP_ROOT = NASDAQ_ROOT / "groups"
REPORTS_ROOT = PROJECT_ROOT / "reports"
CONFIG_ROOT = PROJECT_ROOT / "config"

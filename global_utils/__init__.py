from pathlib import Path
import sys
import importlib

# Resolve GVHMR root from ./GVHMR (relative to this package or cwd)
_THIS_DIR = Path(__file__).resolve().parent
_CWD = Path.cwd().resolve()

GVHMR_ROOT = (_THIS_DIR.parent / "GVHMR").resolve()
if not GVHMR_ROOT.exists():
    GVHMR_ROOT = (_CWD / "GVHMR").resolve()

if not GVHMR_ROOT.exists():
    raise ImportError(
        "GVHMR not found.\n"
        f"Expected at: {_THIS_DIR.parent / 'GVHMR'} or {_CWD / 'GVHMR'}\n"
        f"global_utils dir: {_THIS_DIR}\n"
        f"Current working dir: {_CWD}"
    )

# Make `import hmr4d` work
if str(GVHMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GVHMR_ROOT))

# Alias: global_utils.hmr4d -> hmr4d (from GVHMR)
try:
    _hmr4d = importlib.import_module("hmr4d")
    sys.modules[__name__ + ".hmr4d"] = _hmr4d
    globals()["hmr4d"] = _hmr4d
except Exception as e:
    raise ImportError(
        f"Found GVHMR at {GVHMR_ROOT}, but failed to import `hmr4d`.\n"
        f"Expected package dir: {GVHMR_ROOT / 'hmr4d'}\n"
        f"Original error: {repr(e)}"
    ) from e

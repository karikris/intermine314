"""Bounded fresh interpreters using the same source tree or wheel as the parent."""

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_cold(script, *, timeout=30):
    package = Path(importlib.util.find_spec("intermine314").origin).resolve()
    bootstrap = f'''
import importlib.util, sys
from pathlib import Path
sys.path[:0] = [{str(package.parent.parent)!r}, {str(ROOT)!r}]
expected_package = Path({str(package)!r})
assert Path(importlib.util.find_spec("intermine314").origin).resolve() == expected_package
'''
    verify = '''
for name, module in tuple(sys.modules.items()):
    if (name == 'intermine314' or name.startswith('intermine314.')) and getattr(module, '__file__', None):
        assert Path(module.__file__).resolve().is_relative_to(expected_package.parent), (name, module.__file__)
'''
    try:
        return subprocess.run(
            [sys.executable, "-I", "-c", bootstrap + "\n" + script + "\n" + verify],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"Cold interpreter exceeded {timeout}s and was killed/reaped. "
            f"stdout={error.stdout!r}; stderr={error.stderr!r}"
        ) from error

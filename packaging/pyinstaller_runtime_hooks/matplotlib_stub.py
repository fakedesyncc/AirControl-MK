"""Small matplotlib fallback for the end-user PyInstaller bundle.

MediaPipe imports ``matplotlib.pyplot`` from its drawing helpers during package
initialization, even though AirControl draws its own landmarks. The product
bundle excludes the full matplotlib stack, so this hook supplies the tiny subset
needed for import-time compatibility.
"""

from __future__ import annotations

import sys
import types

try:
    import matplotlib.pyplot  # noqa: F401
except Exception:
    class _NoopAxes:
        def __getattr__(self, _name):
            return self._noop

        def _noop(self, *args, **kwargs):
            return self

    def _figure(*args, **kwargs):
        return _NoopAxes()

    def _axes(*args, **kwargs):
        return _NoopAxes()

    def _show(*args, **kwargs):
        return None

    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = _figure
    pyplot.axes = _axes
    pyplot.show = _show
    matplotlib.pyplot = pyplot
    sys.modules["matplotlib"] = matplotlib
    sys.modules["matplotlib.pyplot"] = pyplot

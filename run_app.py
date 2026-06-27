"""Точка входа для PyInstaller-сборки AirControl.

Запускает тот же CLI, что и `python -m aircontrol` (по умолчанию — приложение).
"""

import multiprocessing
import sys
import traceback

from aircontrol.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()   # безопасно для joblib/sklearn в бандле
    try:
        main()
    except Exception as exc:
        from aircontrol.crash import (
            show_crash_message,
            write_crash_log,
            write_startup_support_bundle,
        )

        path = write_crash_log(exc)
        support_path = write_startup_support_bundle()
        traceback.print_exc()
        show_crash_message(path, exc, support_path)
        sys.exit(1)

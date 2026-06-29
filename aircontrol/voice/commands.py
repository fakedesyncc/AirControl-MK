"""Парсер и исполнитель голосовых команд (кросс-платформенный).

Команды разбиты по группам: буфер обмена, навигация, вкладки, громкость,
управление окном камеры, запуск/закрытие приложений. Системные операции идут
через платформенный бэкенд, ввод/хоткеи — через ActionExecutor. Если фраза не
распознана как команда — текст печатается в активное поле."""

import re
import sys
from typing import Callable, Dict, Optional

from ..control.input_backend import Key


class CommandProcessor:
    def __init__(self, action_executor, window_callbacks: Optional[Dict[str, Callable]] = None):
        self.act = action_executor
        self.platform = action_executor.platform
        self.mod = action_executor.mod
        self.is_mac = sys.platform == "darwin"
        self.window_callbacks = window_callbacks or {}

    def process(self, text: str) -> bool:
        """Возвращает True, если фраза распознана и выполнена как команда."""
        t = text.lower().strip()

        # ---- Буфер обмена ----
        if t in ("копируй", "копировать", "копируй это"):
            self.act.hotkey(self.mod, "c"); return True
        if t in ("вставь", "вставить", "вставь это"):
            self.act.hotkey(self.mod, "v"); return True
        if t in ("вырежи", "вырезать", "вырежь"):
            self.act.hotkey(self.mod, "x"); return True

        # ---- Навигация ----
        if t in ("назад", "назад страницу", "предыдущая страница"):
            self._nav_back(); return True
        if t in ("вперёд", "вперед", "следующая страница"):
            self._nav_forward(); return True
        if any(w in t for w in ("обнови", "обновить", "перезагрузи")):
            self.act.hotkey(self.mod, "r"); return True

        # ---- Вкладки ----
        if any(w in t for w in ("следующая вкладка", "вкладка вперёд")):
            # Ctrl+Tab переключает вкладку в браузерах на всех ОС (на macOS
            # Cmd+Tab — это переключатель приложений, а не вкладок).
            self.act.hotkey(Key.ctrl, Key.tab); return True
        if any(w in t for w in ("предыдущая вкладка", "вкладка назад")):
            self.act.hotkey(Key.ctrl, Key.shift, Key.tab); return True
        if any(w in t for w in ("закрой вкладку", "закрыть вкладку")):
            self.act.hotkey(self.mod, "w"); return True
        if any(w in t for w in ("новая вкладка", "новый таб")):
            self.act.hotkey(self.mod, "t"); return True

        # ---- Громкость ----
        if any(w in t for w in ("громче", "увеличь громкость", "добавь громкость")):
            self.platform.change_volume(10); return True
        if any(w in t for w in ("тише", "уменьши громкость", "убавь громкость")):
            self.platform.change_volume(-10); return True
        if any(w in t for w in ("беззвучно", "выключи звук", "мьют")):
            self.platform.set_muted(True); return True
        if any(w in t for w in ("включи звук", "снять беззвучье")):
            self.platform.set_muted(False); return True

        # ---- Окно камеры ----
        if any(w in t for w in ("сверни камеру", "спрячь камеру", "минимизируй камеру")):
            cb = self.window_callbacks.get("minimize")
            if cb:
                cb()
            return True
        if any(w in t for w in ("покажи камеру", "разверни камеру", "верни камеру")):
            cb = self.window_callbacks.get("restore")
            if cb:
                cb()
            return True

        # ---- YouTube / URL ----
        if t in ("ютуб", "youtube", "открой ютуб"):
            self.platform.open_url("https://www.youtube.com"); return True

        # ---- Приложения ----
        for pattern in (r"открой\s+(.+)", r"запусти\s+(.+)", r"включи\s+(.+)"):
            m = re.search(pattern, t)
            if m:
                self.platform.open_app(m.group(1).strip()); return True
        for pattern in (r"закрой\s+(.+)", r"выключи\s+(.+)"):
            m = re.search(pattern, t)
            if m:
                self.platform.close_app(m.group(1).strip()); return True
        for pattern in (r"сверни\s+(.+)", r"минимизируй\s+(.+)", r"скрой\s+(.+)"):
            m = re.search(pattern, t)
            if m:
                self.platform.minimize_app(m.group(1).strip()); return True

        return False

    def _nav_back(self) -> None:
        if self.is_mac:
            self.act.hotkey(self.mod, "[")
        else:
            self.act.hotkey(Key.alt, Key.left)

    def _nav_forward(self) -> None:
        if self.is_mac:
            self.act.hotkey(self.mod, "]")
        else:
            self.act.hotkey(Key.alt, Key.right)

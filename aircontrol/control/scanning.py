"""Конечный автомат сканирующей клавиатуры (row–column scanning).

Это «чистый» модуль без Tk/mediapipe/камеры: только логика подсветки и выбора.
Такой автомат — классический паттерн AAC для пользователей с тяжёлыми
двигательными нарушениями, которым недоступны щипки и точное наведение. Всё
управление сводится к ОДНОМУ событию «выбор» (dwell/удержание или клавиша):

  1. подсветка автоматически перебирает строки (режим ROW);
  2. «выбор» фиксирует текущую строку и переключает перебор на клавиши этой
     строки (режим COL);
  3. ещё один «выбор» отправляет клавишу наружу через callback и возвращает
     автомат к перебору строк.

Вынесение логики в отдельный модуль делает её полностью юнит-тестируемой без
GUI и тяжёлых зависимостей: представление (aircontrol/ui/scanning_keyboard.py)
лишь рисует раскладку, гонит таймер tick() и пробрасывает «выбор» в select().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional


class ScanMode(Enum):
    """Режим перебора подсветки."""

    ROW = "row"   # перебираем строки целиком
    COL = "col"   # перебираем клавиши внутри выбранной строки


# Типы клавиш (action token), которые автомат отдаёт наружу при выборе.
KEY_CHAR = "char"            # печатаемый символ (label попадает в текст)
KEY_SPACE = "space"          # пробел
KEY_BACKSPACE = "backspace"  # удалить символ слева
KEY_ENTER = "enter"          # перевод строки / подтверждение
KEY_EXIT = "exit"            # выход из меню — останавливает сканирование


@dataclass(frozen=True)
class ScanKey:
    """Одна клавиша раскладки.

    label  — что показывать на кнопке;
    kind   — тип клавиши (см. KEY_* выше);
    value  — символ для печати (для KEY_CHAR; по умолчанию совпадает с label).
    """

    label: str
    kind: str = KEY_CHAR
    value: Optional[str] = None

    def char(self) -> str:
        """Символ для вставки в текст (актуально для KEY_CHAR)."""
        return self.value if self.value is not None else self.label


@dataclass
class ScanOutput:
    """Результат выбора клавиши, передаваемый в callback.

    kind   — тип клавиши (KEY_*);
    label  — подпись клавиши (для логов/озвучки);
    char   — символ для печати (пусто для не-символьных клавиш).
    """

    kind: str
    label: str
    char: str = ""


def default_layout() -> List[List[ScanKey]]:
    """Компактная QWERTY-подобная раскладка + служебные клавиши.

    Последняя строка — служебная: пробел, удаление, ввод и выход из меню.
    Выход (KEY_EXIT) останавливает сканирование, чтобы пользователь всегда мог
    закрыть клавиатуру одним «выбором», не нажимая ничего лишнего.
    """
    rows_text = ["qwertyu", "iopasdf", "ghjklzx", "cvbnm"]
    layout: List[List[ScanKey]] = [
        [ScanKey(ch) for ch in row] for row in rows_text
    ]
    layout.append([
        ScanKey("␣", KEY_SPACE),
        ScanKey("⌫", KEY_BACKSPACE),
        ScanKey("⏎", KEY_ENTER),
        ScanKey("✕", KEY_EXIT),
    ])
    return layout


@dataclass
class ScanKeyboard:
    """Конечный автомат сканирующей клавиатуры.

    layout      — строки клавиш (по умолчанию default_layout());
    on_output   — callback, вызывается с ScanOutput при выборе клавиши;
    on_exit     — callback без аргументов при выборе клавиши выхода (KEY_EXIT);
    max_loops   — сколько полных проходов подсветки сделать без выбора, прежде
                  чем остановить сканирование (0 = не останавливаться).

    Состояние:
        mode    — ROW или COL;
        row     — индекс активной строки в режиме COL;
        index   — текущая подсвеченная позиция в current_targets();
        loops   — счётчик завершённых проходов в текущем режиме;
        running — активно ли сканирование (False — подсветка стоит).
    """

    layout: List[List[ScanKey]] = field(default_factory=default_layout)
    on_output: Optional[Callable[[ScanOutput], None]] = None
    on_exit: Optional[Callable[[], None]] = None
    max_loops: int = 0

    mode: ScanMode = field(default=ScanMode.ROW, init=False)
    row: int = field(default=0, init=False)
    index: int = field(default=0, init=False)
    loops: int = field(default=0, init=False)
    running: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.layout or any(len(r) == 0 for r in self.layout):
            raise ValueError("layout должен содержать непустые строки клавиш")
        self.reset()

    # ---- состояние / запросы ----------------------------------------------

    def reset(self) -> None:
        """Сбросить автомат к перебору строк с начала и включить сканирование."""
        self.mode = ScanMode.ROW
        self.row = 0
        self.index = 0
        self.loops = 0
        self.running = True

    def current_targets(self) -> List:
        """То, что сейчас перебирается: список строк (ROW) или клавиш строки (COL)."""
        if self.mode is ScanMode.ROW:
            return self.layout
        return self.layout[self.row]

    def current_key(self) -> Optional[ScanKey]:
        """Подсвеченная клавиша в режиме COL (в режиме ROW — None)."""
        if self.mode is ScanMode.COL:
            return self.layout[self.row][self.index]
        return None

    def current_row_keys(self) -> List[ScanKey]:
        """Клавиши текущей подсвеченной/выбранной строки."""
        row = self.index if self.mode is ScanMode.ROW else self.row
        return self.layout[row]

    # ---- переходы ----------------------------------------------------------

    def tick(self) -> None:
        """Сдвинуть подсветку на одну позицию вперёд (по таймеру представления).

        При проходе через конец списка индекс заворачивается на начало, а
        счётчик loops увеличивается. Если задан max_loops и достигнут лимит без
        выбора — сканирование останавливается (защита от бесконечного перебора)."""
        if not self.running:
            return
        size = len(self.current_targets())
        self.index += 1
        if self.index >= size:
            self.index = 0
            self.loops += 1
            if self.max_loops and self.loops >= self.max_loops:
                self.running = False

    def select(self) -> Optional[ScanOutput]:
        """Обработать единственное событие «выбор».

        ROW → зафиксировать строку и перейти к перебору её клавиш (COL).
        COL → отдать выбранную клавишу в on_output и вернуться к перебору строк;
              клавиша выхода (KEY_EXIT) останавливает сканирование и зовёт on_exit.

        Возвращает ScanOutput, если клавиша была отправлена, иначе None."""
        if not self.running:
            return None
        if self.mode is ScanMode.ROW:
            self.row = self.index
            self.mode = ScanMode.COL
            self.index = 0
            self.loops = 0
            return None
        return self._emit_current_key()

    def _emit_current_key(self) -> Optional[ScanOutput]:
        key = self.layout[self.row][self.index]
        if key.kind == KEY_EXIT:
            self.running = False
            if self.on_exit is not None:
                self.on_exit()
            return ScanOutput(kind=KEY_EXIT, label=key.label, char="")

        char = key.char() if key.kind == KEY_CHAR else ""
        output = ScanOutput(kind=key.kind, label=key.label, char=char)
        if self.on_output is not None:
            self.on_output(output)

        # Вернуться к перебору строк для следующего символа.
        self.mode = ScanMode.ROW
        self.index = 0
        self.loops = 0
        return output

    def stop(self) -> None:
        """Остановить сканирование (подсветка замирает)."""
        self.running = False

    def start(self) -> None:
        """Возобновить сканирование, не сбрасывая позицию."""
        self.running = True

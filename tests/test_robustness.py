"""Тесты устойчивости к «залипшим» модификаторам/кнопкам (stdlib unittest).

Запуск:  python -m unittest tests.test_robustness
Проверяют, что release_all() надёжно отпускает любые удержанные кнопки мыши и
клавиши-модификаторы, идемпотентен и не бросает исключений даже при сбойном
бэкенде. Реальный pynput не используется — подставляем фейковый бэкенд,
записывающий вызовы press/release.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.control.actions import ActionExecutor
from aircontrol.control.input_backend import Button, Key


class RecordingKeyboard:
    """Фейковая клавиатура: помнит порядок press/release."""

    def __init__(self):
        self.pressed = []
        self.released = []

    def press(self, key):
        self.pressed.append(key)

    def release(self, key):
        self.released.append(key)

    def type(self, text):
        pass


class RecordingMouse:
    """Фейковая мышь: помнит press/release кнопок и позицию."""

    def __init__(self):
        self.position = (0, 0)
        self.pressed = []
        self.released = []

    def press(self, button):
        self.pressed.append(button)

    def release(self, button):
        self.released.append(button)

    def click(self, button, count=1):
        pass

    def scroll(self, dx, dy):
        pass


class ExplodingKeyboard(RecordingKeyboard):
    """Клавиатура, у которой release всегда падает."""

    def release(self, key):
        super().release(key)
        raise RuntimeError("keyboard release failed")


class ExplodingMouse(RecordingMouse):
    """Мышь, у которой release всегда падает."""

    def release(self, button):
        super().release(button)
        raise RuntimeError("mouse release failed")


def _make_executor(tmp_dir, keyboard, mouse):
    """ActionExecutor с подставленным (не настоящим) бэкендом.

    Строим в dry_run, чтобы не трогать реальный pynput при создании, затем
    переключаем dry_run=False и подменяем контроллеры на фейковые — так же,
    как это делают тесты в tests/test_core.py.
    """
    act = ActionExecutor(tmp_dir, dry_run=True)
    act.dry_run = False
    act.keyboard = keyboard
    act.mouse = mouse
    act.mod = Key.ctrl
    return act


class TestReleaseAllRobustness(unittest.TestCase):
    def test_drag_then_release_all_releases_left_button(self):
        # После зажатия ЛКМ (drag) release_all должен её отпустить.
        with tempfile.TemporaryDirectory() as d:
            kb, ms = RecordingKeyboard(), RecordingMouse()
            act = _make_executor(d, kb, ms)
            act.left_down()
            self.assertIn(Button.left, ms.pressed)
            self.assertTrue(act._left_down)

            act.release_all()
            self.assertIn(Button.left, ms.released)
            self.assertFalse(act._left_down)

    def test_release_all_clears_held_modifier_from_interrupted_combo(self):
        # Имитируем прерванное комбо: модификатор «остался зажат» в учёте.
        with tempfile.TemporaryDirectory() as d:
            kb, ms = RecordingKeyboard(), RecordingMouse()
            act = _make_executor(d, kb, ms)
            act.keyboard.press(Key.cmd)
            act._held_keys.add(Key.cmd)

            act.release_all()
            self.assertIn(Key.cmd, kb.released)
            self.assertEqual(act._held_keys, set())

    def test_release_all_releases_known_modifiers_defensively(self):
        # Даже без учёта held release_all обязан отпустить базовые модификаторы.
        with tempfile.TemporaryDirectory() as d:
            kb, ms = RecordingKeyboard(), RecordingMouse()
            act = _make_executor(d, kb, ms)

            act.release_all()
            for mod in (Key.ctrl, Key.cmd, Key.shift, Key.alt):
                self.assertIn(mod, kb.released,
                              f"модификатор {mod} должен быть отпущен")

    def test_release_all_releases_all_mouse_buttons(self):
        # Все кнопки мыши отпускаются на всякий случай (правая/средняя тоже).
        with tempfile.TemporaryDirectory() as d:
            kb, ms = RecordingKeyboard(), RecordingMouse()
            act = _make_executor(d, kb, ms)

            act.release_all()
            for button in (Button.left, Button.right, Button.middle):
                self.assertIn(button, ms.released,
                              f"кнопка {button} должна быть отпущена")

    def test_hotkey_failure_then_release_all_clears_modifier(self):
        # Полный сценарий: hotkey с падающим press оставляет зажатый модификатор?
        # Сначала hotkey сам пытается всё отпустить, а release_all добивает.
        with tempfile.TemporaryDirectory() as d:
            ms = RecordingMouse()

            class HalfBrokenKeyboard(RecordingKeyboard):
                def press(self, key):
                    super().press(key)
                    if key == "x":
                        raise RuntimeError("press x failed")

            kb = HalfBrokenKeyboard()
            act = _make_executor(d, kb, ms)
            act.hotkey(Key.cmd, "x")
            # hotkey уже отпустил то, что успел зажать, и почистил учёт.
            self.assertEqual(act._held_keys, set())

            act.release_all()
            # Командный модификатор всё равно отпускается (страховочный набор).
            self.assertIn(Key.cmd, kb.released)

    def test_release_all_is_idempotent(self):
        # Повторный вызов безопасен и не бросает.
        with tempfile.TemporaryDirectory() as d:
            kb, ms = RecordingKeyboard(), RecordingMouse()
            act = _make_executor(d, kb, ms)
            act.left_down()

            act.release_all()
            first_state = (act._left_down, set(act._held_keys))
            try:
                act.release_all()
            except Exception as exc:  # pragma: no cover - тест должен поймать падение
                self.fail(f"повторный release_all бросил исключение: {exc}")
            self.assertEqual((act._left_down, set(act._held_keys)), first_state)
            self.assertFalse(act._left_down)
            self.assertEqual(act._held_keys, set())

    def test_release_all_swallows_keyboard_backend_errors(self):
        # Сбой keyboard.release не должен пробрасываться наружу.
        with tempfile.TemporaryDirectory() as d:
            kb, ms = ExplodingKeyboard(), RecordingMouse()
            act = _make_executor(d, kb, ms)
            act._held_keys.add(Key.cmd)

            try:
                act.release_all()
            except Exception as exc:  # pragma: no cover
                self.fail(f"release_all пробросил ошибку клавиатуры: {exc}")
            # Учёт всё равно очищен, а ошибка зафиксирована.
            self.assertEqual(act._held_keys, set())
            self.assertIn("release_all", act.last_input_error)
            self.assertGreater(act.input_error_count, 0)

    def test_release_all_swallows_mouse_backend_errors(self):
        # Сбой mouse.release не должен пробрасываться наружу.
        with tempfile.TemporaryDirectory() as d:
            kb, ms = RecordingKeyboard(), ExplodingMouse()
            act = _make_executor(d, kb, ms)
            act.left_down()  # учтём drag, чтобы left_up попытался отпустить ЛКМ

            try:
                act.release_all()
            except Exception as exc:  # pragma: no cover
                self.fail(f"release_all пробросил ошибку мыши: {exc}")
            self.assertFalse(act._left_down)
            # Несмотря на падения мыши, клавиатура-страховка отработала.
            self.assertIn(Key.ctrl, kb.released)
            self.assertGreater(act.input_error_count, 0)

    def test_release_all_in_dry_run_does_not_touch_backends(self):
        # В dry_run release_all не должен дёргать реальные контроллеры.
        with tempfile.TemporaryDirectory() as d:
            act = ActionExecutor(d, dry_run=True)
            kb, ms = RecordingKeyboard(), RecordingMouse()
            act.keyboard = kb
            act.mouse = ms
            act.left_down()  # в dry_run просто помечает _left_down

            act.release_all()
            self.assertFalse(act._left_down)
            self.assertEqual(kb.released, [])
            self.assertEqual(ms.released, [])


if __name__ == "__main__":
    unittest.main()

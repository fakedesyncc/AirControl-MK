"""Тесты координатора слияния модальностей (stdlib unittest, без камеры/ОС/голоса).

Запуск:  python -m unittest tests.test_fusion

Покрывают логику MultimodalCoordinator.process() на фейках исполнителя действий,
контроллера курсора и распознавателя голоса. MediaPipe не нужен — координатор его
не тянет; используем настоящие FrameGestures, чтобы тесты отражали реальный
датакласс, а не его подделку.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.config import FusionConfig
from aircontrol.fusion.coordinator import MultimodalCoordinator
from aircontrol.gestures.engine import FrameGestures, GestureEvent


class FakeAction:
    """Исполнитель действий: записывает вызовы вместо реального ввода в ОС."""

    def __init__(self):
        self.executed = []      # имена дискретных действий
        self.scrolls = []       # величины шагов скролла
        self.released = 0       # сколько раз отпускали зажатые кнопки

    def execute(self, action):
        self.executed.append(action)

    def scroll(self, steps):
        self.scrolls.append(steps)

    def release_all(self):
        self.released += 1


class FakeCursor:
    """Контроллер курсора: считает обновления и возвращает заданный dwell-клик."""

    def __init__(self, dwell_action=None):
        self.updates = []           # (fg, timestamp) каждого вызова update
        self._dwell_action = dwell_action

    def update(self, fg, timestamp):
        self.updates.append((fg, timestamp))
        return self._dwell_action


class FakeVoice:
    """Распознаватель голоса: фиксирует старт прослушивания, без микрофона."""

    def __init__(self, is_listening=False, last_status=""):
        self.is_listening = is_listening
        self.last_status = last_status
        self.start_calls = 0

    def start_listening(self):
        self.start_calls += 1
        self.is_listening = True


def make_coordinator(cfg=None, action=None, cursor=None, voice=None):
    """Собирает координатор с фейками; cfg по умолчанию — дефолтный FusionConfig."""
    cfg = cfg if cfg is not None else FusionConfig()
    action = action if action is not None else FakeAction()
    cursor = cursor if cursor is not None else FakeCursor()
    voice = voice if voice is not None else FakeVoice()
    coord = MultimodalCoordinator(cfg, action, cursor, voice)
    return coord, cfg, action, cursor, voice


class TestDiscreteEvents(unittest.TestCase):
    def test_events_dispatched_and_scroll_mode_skipped(self):
        # FUSE: дискретные события из fg.events уходят исполнителю, кроме scroll_mode.
        coord, _cfg, action, _cursor, _voice = make_coordinator()
        fg = FrameGestures(
            hand_detected=True,
            cursor_norm=(0.5, 0.5),
            events=[
                GestureEvent("left_down"),
                GestureEvent("scroll_mode"),   # служебное — не действие
                GestureEvent("left_up"),
            ],
        )
        status = coord.process(fg, timestamp=1.0)
        self.assertEqual(action.executed, ["left_down", "left_up"])
        self.assertNotIn("scroll_mode", action.executed)
        self.assertEqual(status.fired_actions, ["left_down", "left_up"])


class TestScrollAccumulation(unittest.TestCase):
    def test_fractional_scroll_accumulates_then_fires_whole_steps(self):
        # FUSE: дробный scroll_delta копится между кадрами; срабатывают целые шаги.
        coord, _cfg, action, _cursor, _voice = make_coordinator()

        # 0.4 < 1.0 — пока ничего не уходит в скролл.
        coord.process(FrameGestures(scroll_delta=0.4), timestamp=1.0)
        self.assertEqual(action.scrolls, [])

        # 0.4 + 0.4 = 0.8 < 1.0 — всё ещё копим.
        coord.process(FrameGestures(scroll_delta=0.4), timestamp=2.0)
        self.assertEqual(action.scrolls, [])

        # 0.8 + 0.4 = 1.2 → один целый шаг, остаток 0.2 остаётся в накопителе.
        status = coord.process(FrameGestures(scroll_delta=0.4), timestamp=3.0)
        self.assertEqual(action.scrolls, [1])
        self.assertIn("scroll:1", status.fired_actions)
        self.assertAlmostEqual(coord._scroll_accum, 0.2)

    def test_scroll_resets_when_scrolling_stops(self):
        # FUSE: пауза в скролле обнуляет накопитель (нет фантомного «дошага»).
        coord, _cfg, action, _cursor, _voice = make_coordinator()

        coord.process(FrameGestures(scroll_delta=0.7), timestamp=1.0)
        self.assertEqual(action.scrolls, [])
        self.assertAlmostEqual(coord._scroll_accum, 0.7)

        # Кадр без скролла — накопитель сбрасывается.
        coord.process(FrameGestures(scroll_delta=0.0), timestamp=2.0)
        self.assertEqual(action.scrolls, [])
        self.assertEqual(coord._scroll_accum, 0.0)

    def test_negative_scroll_fires_negative_steps(self):
        # FUSE: скролл вверх (отрицательная дельта) тоже накапливается и срабатывает.
        coord, _cfg, action, _cursor, _voice = make_coordinator()
        coord.process(FrameGestures(scroll_delta=-0.6), timestamp=1.0)
        status = coord.process(FrameGestures(scroll_delta=-0.6), timestamp=2.0)
        self.assertEqual(action.scrolls, [-1])
        self.assertIn("scroll:-1", status.fired_actions)


class TestVoiceGating(unittest.TestCase):
    def test_cursor_suppressed_while_listening(self):
        # FUSE: пока слушаем голос и включено подавление — курсор не обновляем.
        cfg = FusionConfig()
        cfg.suppress_cursor_while_listening = True
        voice = FakeVoice(is_listening=True, last_status="listening")
        coord, _cfg, _action, cursor, voice = make_coordinator(cfg=cfg, voice=voice)

        fg = FrameGestures(hand_detected=True, cursor_norm=(0.5, 0.5))
        status = coord.process(fg, timestamp=1.0)

        self.assertEqual(cursor.updates, [])   # курсор не тронут в этот кадр
        self.assertTrue(status.listening)

    def test_cursor_updated_when_suppression_disabled(self):
        # FUSE (контроль): при выключенном подавлении курсор обновляется даже слушая.
        cfg = FusionConfig()
        cfg.suppress_cursor_while_listening = False
        voice = FakeVoice(is_listening=True)
        coord, _cfg, _action, cursor, _voice = make_coordinator(cfg=cfg, voice=voice)

        fg = FrameGestures(hand_detected=True, cursor_norm=(0.5, 0.5))
        coord.process(fg, timestamp=1.0)
        self.assertEqual(len(cursor.updates), 1)

    def test_fist_gesture_starts_listening(self):
        # FUSE: жест «кулак» (listening_requested) запускает прослушивание.
        voice = FakeVoice(is_listening=False)
        coord, _cfg, _action, _cursor, voice = make_coordinator(voice=voice)

        fg = FrameGestures(listening_requested=True)
        status = coord.process(fg, timestamp=1.0)

        self.assertEqual(voice.start_calls, 1)
        self.assertTrue(voice.is_listening)
        self.assertTrue(status.listening)

    def test_fist_does_not_restart_when_already_listening(self):
        # FUSE: повторный кулак во время прослушивания не дёргает start заново.
        voice = FakeVoice(is_listening=True)
        coord, _cfg, _action, _cursor, voice = make_coordinator(voice=voice)

        coord.process(FrameGestures(listening_requested=True), timestamp=1.0)
        self.assertEqual(voice.start_calls, 0)


class TestDwellAction(unittest.TestCase):
    def test_dwell_action_executed_via_action_executor(self):
        # FUSE: dwell-клик, возвращённый cursor.update, исполняется исполнителем.
        cursor = FakeCursor(dwell_action="left_click")
        coord, _cfg, action, cursor, _voice = make_coordinator(cursor=cursor)

        fg = FrameGestures(hand_detected=True, cursor_norm=(0.5, 0.5))
        status = coord.process(fg, timestamp=1.0)

        self.assertEqual(len(cursor.updates), 1)
        self.assertEqual(action.executed, ["left_click"])
        self.assertIn("dwell:left_click", status.fired_actions)

    def test_no_dwell_action_means_no_execute(self):
        # FUSE (контроль): без dwell-клика исполнитель действий не вызывается.
        cursor = FakeCursor(dwell_action=None)
        coord, _cfg, action, cursor, _voice = make_coordinator(cursor=cursor)

        coord.process(FrameGestures(hand_detected=True, cursor_norm=(0.5, 0.5)),
                      timestamp=1.0)
        self.assertEqual(len(cursor.updates), 1)
        self.assertEqual(action.executed, [])


class TestShutdown(unittest.TestCase):
    def test_shutdown_releases_all(self):
        # FUSE: shutdown() отпускает зажатые кнопки через release_all().
        coord, _cfg, action, _cursor, _voice = make_coordinator()
        coord.shutdown()
        self.assertEqual(action.released, 1)


if __name__ == "__main__":
    unittest.main()

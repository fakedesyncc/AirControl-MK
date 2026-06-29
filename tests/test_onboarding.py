"""First-run onboarding logic tests.

These cover the *pure* helpers that drive the first-run wizard wording and the
"obvious next action" recommendation for a non-technical, accessibility-first
audience. No Tk window is constructed here: only the decision/text helpers are
exercised, so the suite runs headless.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aircontrol.launcher import (
    FIRST_RUN_CHECKING_HINT,
    SAFE_START_REASSURANCE,
    build_first_run_status,
    first_run_recommended_action,
)


# Doctor-report fragments reused across cases. They mirror the real strings the
# diagnostics module emits and that build_first_run_status parses.
READY_REPORT = (
    "OpenCV: OK\nMediaPipe: OK\nHand model: OK (/tmp/model)\nTkinter: OK\n"
    "Camera scan: 0..1\ncamera[0]: OK frame=(360, 480, 3)\n"
    "input backend: OK (pynput)\n"
    "input probe: OK (backend=pynput; mouse_move=requested; backend initialized)\n"
    "input mouse move probe: OK (mouse moved and was restored near the original position)\n"
)
CAMERA_MISSING_REPORT = (
    "OpenCV: OK\nMediaPipe: OK\nHand model: OK (/tmp/model)\nTkinter: OK\n"
    "Camera scan: 0..1\ncamera[0]: not available\n"
    "input backend: OK (pynput)\ninput probe: OK (backend=pynput)\n"
)
INPUT_PROBE_SKIPPED_REPORT = (
    "OpenCV: OK\nMediaPipe: OK\nHand model: OK (/tmp/model)\nTkinter: OK\n"
    "Camera scan: 0..1\ncamera[0]: OK frame=(360, 480, 3)\n"
    "input backend: OK (ydotool)\n"
    "input probe: WARN (backend=ydotool; mouse_move=requested; visible movement skipped)\n"
    "input mouse move probe: SKIPPED (ydotool cannot safely read the current cursor position)\n"
)


class TestSafeReassuranceText(unittest.TestCase):
    def test_start_reassurance_promises_no_clicks(self):
        # The very first thing the user reads must make the safe path explicit.
        self.assertIn("безопасн", SAFE_START_REASSURANCE.lower())
        self.assertIn("НЕ отправляются", SAFE_START_REASSURANCE)

    def test_checking_hint_is_non_empty(self):
        self.assertTrue(FIRST_RUN_CHECKING_HINT.strip())


class TestFirstRunRecommendedAction(unittest.TestCase):
    def test_ready_recommends_safe_training(self):
        statuses = build_first_run_status(READY_REPORT, ["Status: ready for safe training."])
        action = first_run_recommended_action(statuses)
        self.assertEqual(action["action"], "safe_training")
        self.assertEqual(action["status"], "ok")
        # Headline must point at the safe button and reassure about clicks.
        self.assertIn("безопасн", action["headline"].lower())
        self.assertIn("не отправля", action["headline"].lower())
        # And it should hand off toward calibration before real control.
        self.assertIn("калибров", action["headline"].lower())

    def test_camera_missing_recommends_diagnostics(self):
        statuses = build_first_run_status(
            CAMERA_MISSING_REPORT,
            ["Status: needs attention before assistive control."],
        )
        action = first_run_recommended_action(statuses)
        self.assertEqual(action["action"], "diagnostics")
        self.assertEqual(action["status"], "fail")
        # A blocking problem must NOT nudge the user into real control.
        self.assertIn("диагностик", action["headline"].lower())

    def test_input_probe_skipped_recommends_cautious_safe_training(self):
        statuses = build_first_run_status(
            INPUT_PROBE_SKIPPED_REPORT,
            ["Status: ready for safe training."],
        )
        action = first_run_recommended_action(statuses)
        # A warning is not blocking: still steer to the safe path first, but
        # the headline should be cautious about enabling real control.
        self.assertEqual(action["action"], "safe_training")
        self.assertEqual(action["status"], "warn")
        self.assertIn("безопасн", action["headline"].lower())

    def test_missing_next_status_defaults_to_safe(self):
        # Defensive: if the next step is somehow absent, never block the user.
        action = first_run_recommended_action(
            [{"id": "camera", "status": "ok", "message": "ok"}]
        )
        self.assertEqual(action["action"], "safe_training")
        self.assertEqual(action["status"], "ok")

    def test_empty_statuses_defaults_to_safe(self):
        action = first_run_recommended_action([])
        self.assertEqual(action["action"], "safe_training")

    def test_headline_always_present_and_nonempty(self):
        for report, summary in [
            (READY_REPORT, ["Status: ready for safe training."]),
            (CAMERA_MISSING_REPORT, ["Status: needs attention before assistive control."]),
            (INPUT_PROBE_SKIPPED_REPORT, ["Status: ready for safe training."]),
        ]:
            statuses = build_first_run_status(report, summary)
            action = first_run_recommended_action(statuses)
            self.assertTrue(action["headline"].strip())
            self.assertIn(action["action"], {"safe_training", "diagnostics"})


class TestRecommendationMatchesNextStatus(unittest.TestCase):
    """The recommendation must stay consistent with build_first_run_status."""

    def test_diagnostics_only_when_next_is_fail(self):
        for status_value, expected_action in [
            ("ok", "safe_training"),
            ("warn", "safe_training"),
            ("fail", "diagnostics"),
        ]:
            statuses = [{"id": "next", "status": status_value, "message": "x"}]
            action = first_run_recommended_action(statuses)
            self.assertEqual(action["action"], expected_action)
            self.assertEqual(action["status"], status_value)


if __name__ == "__main__":
    unittest.main()

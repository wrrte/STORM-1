"""Regression checks for RAM binding corruption, unsigned math, and missing masks.

Run: python -m unittest probing.test_ram_states -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from probing.ram_states import extract_state, read_ale_ram, validate_ram
from probing.verify_ram_states import load_data, select_frames, write_report


def blank_ram():
    ram = np.zeros(128, np.uint8)
    ram[127] = 1  # distinguish a legitimate fixture from the constant-RAM binding bug
    return ram


class RamStatesTest(unittest.TestCase):
    def test_reject_corrupt_constant_ram(self):
        for value in (0, 67, 192, 255):
            with self.assertRaisesRegex(ValueError, "identical"):
                validate_ram(np.full(128, value, np.uint8))

    def test_buffer_overload_avoids_old_binding_bug(self):
        expected = np.arange(128, dtype=np.uint8)

        class OldALE:
            def getRAM(self, target=None):
                if target is None:
                    return np.full(128, 67, np.uint8)
                target[:] = expected

        np.testing.assert_array_equal(read_ale_ram(OldALE()), expected)

    def test_fallback_to_modern_no_argument_api(self):
        expected = np.arange(128, dtype=np.uint8)

        class ModernALE:
            def getRAM(self):
                return expected

        got = read_ale_ram(ModernALE())
        got[0] = 99
        self.assertEqual(expected[0], 0)  # never alias emulator memory

    def test_seaquest_shared_slot_changes_kind_without_stale_diver(self):
        ram = blank_ram()
        ram[71] = 100
        diver = extract_state("seaquest", ram)
        ram[89] = 4
        missile = extract_state("seaquest", ram)
        self.assertEqual(diver["scalars"]["visible_divers"], 1)
        self.assertEqual(missile["scalars"]["visible_divers"], 0)
        self.assertIn("enemy_missile", [o["kind"] for o in missile["objects"]])
        ram[89] = 0
        self.assertEqual(diver, extract_state("seaquest", ram))

    def test_frostbite_absent_igloo_and_unknown_codes(self):
        ram = blank_ram()
        ram[77] = 255
        state = extract_state("frostbite", ram)
        self.assertEqual(state["scalars"]["igloo_stage"], 0)
        self.assertNotIn("igloo", [o["kind"] for o in state["objects"]])
        ram[77] = 128
        state = extract_state("frostbite", ram)
        self.assertIsNone(state["scalars"]["igloo_stage"])
        self.assertTrue(state["warnings"])

    def test_unsigned_y_arithmetic_does_not_wrap_to_255(self):
        ram = blank_ram()
        ram[36], ram[93] = 4, 0  # shark bobbing offset is -4
        state = extract_state("seaquest", ram)
        shark = next(o for o in state["objects"] if o["kind"] == "shark")
        self.assertLess(shark["xywh"][1], 141)

    def test_bcd_and_missing_ball(self):
        ram = blank_ram()
        ram[18], ram[19] = 0x24, 0x10
        scores = extract_state("boxing", ram)["scalars"]
        self.assertEqual((scores["player_score"], scores["enemy_score"]), (24, 10))
        self.assertNotIn("ball", [o["id"] for o in extract_state("pong", ram)["objects"]])

    def test_report_preserves_absence_mask_and_frame_indices(self):
        ram = np.stack([blank_ram(), blank_ram()])
        ram[1, 49], ram[1, 54] = 100, 100
        data = dict(obs=np.zeros((2, 64, 64, 3), np.uint8), ram=ram)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_report("pong", data, output, 2, 7)
            with np.load(output / "features.npz") as features:
                names = features["names"].tolist()
                x = names.index("ball.x")
                present = names.index("ball.present")
                self.assertTrue(np.isnan(features["values"][0, x]))
                self.assertFalse(features["valid"][0, x])
                self.assertEqual(features["values"][0, present], 0)
                self.assertEqual(features["values"][1, present], 1)
            payload = json.loads((output / "frames/frame_000001.json").read_text())
            self.assertEqual(payload["ram"], ram[1].tolist())
            self.assertTrue((output / "report.html").exists())

    def test_archive_length_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npz"
            np.savez(path, obs=np.zeros((2, 64, 64, 3), np.uint8), ram=blank_ram()[None])
            with self.assertRaisesRegex(ValueError, "aligned"):
                load_data(path, "pong")

    def test_sampling_is_unique_reproducible_and_bounded(self):
        states = [extract_state("pong", blank_ram()) for _ in range(17)]
        first = select_frames(states, 12, 7)
        self.assertEqual(first, select_frames(states, 12, 7))
        self.assertEqual(len(first), 12)
        self.assertTrue(all(0 <= i < len(states) for i in first))
        self.assertEqual(len(select_frames(states, 100, 7)), len(states))


if __name__ == "__main__":
    unittest.main()

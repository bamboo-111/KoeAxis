import sys
import unittest
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "server"))
import progress_snapshot as ps  # noqa: E402

T0 = 1_000_000.0


def progress(done, status="running", updated=None):
    stamp = "2026-01-01T00:00:00+00:00"
    return {"done": done, "total": 12, "status": status, "updated_at": stamp if updated is None else updated}


def at(done, stage_elapsed, factor=1.0, status="running"):
    exp = ps.DEFAULT_SECONDS
    finished_at = ps.STARTUP_SECONDS + sum(exp[s] for s in ps.STAGES[:done]) * factor
    started = ps.parse_time("2026-01-01T00:00:00+00:00") - finished_at
    now = started + finished_at + stage_elapsed
    return ps.build_snapshot(progress(done, status), dict(exp), started, now)


class ProgressSnapshotTest(unittest.TestCase):
    def test_running_stage_is_next_after_done(self):
        snap = at(6, 10)
        self.assertEqual(snap["current"]["name"], "translate")
        self.assertEqual((snap["current"]["index"], snap["current"]["total"]), (7, 12))

    def test_on_pace_has_unit_speed_and_consistent_totals(self):
        snap = at(4, 100)
        self.assertEqual(snap["estimate"]["speed_factor"], 1.0)
        cur = snap["current"]
        self.assertEqual(cur["name"], "speaker")
        self.assertAlmostEqual(cur["percent"], 100 * 100 / 766, delta=0.1)
        later = sum(ps.DEFAULT_SECONDS[s] for s in ps.STAGES[5:])
        self.assertAlmostEqual(snap["overall"]["remaining_s"], 766 - 100 + later, delta=2)

    def test_slow_machine_scales_remaining(self):
        snap = at(4, 0, factor=1.5)
        self.assertEqual(snap["estimate"]["speed_factor"], 1.5)
        later = sum(ps.DEFAULT_SECONDS[s] for s in ps.STAGES[4:])
        self.assertAlmostEqual(snap["overall"]["remaining_s"], later * 1.5, delta=2)

    def test_overrun_caps_percent_and_flags(self):
        snap = at(4, 5000)
        self.assertTrue(snap["current"]["overrun"])
        self.assertEqual(snap["current"]["percent"], 99.0)
        self.assertLess(snap["overall"]["percent"], 100)

    def test_completed(self):
        snap = at(12, 0, status="completed")
        self.assertEqual(snap["run_status"], "completed")
        self.assertEqual(snap["overall"]["percent"], 100.0)
        self.assertEqual(snap["overall"]["remaining_s"], 0)
        self.assertIsNone(snap["current"])

    def test_failed_has_no_eta(self):
        snap = ps.build_snapshot(progress(3), dict(ps.DEFAULT_SECONDS), T0, T0 + 500, "failed")
        self.assertEqual(snap["run_status"], "failed")
        self.assertIsNone(snap["overall"]["remaining_s"])
        self.assertEqual(snap["stages"][3]["state"], "failed")

    def test_not_started(self):
        snap = ps.build_snapshot(None, dict(ps.DEFAULT_SECONDS), T0, T0 + 20)
        self.assertEqual(snap["current"]["name"], "prepare")
        self.assertEqual(snap["stage_done"], 0)


class HistoryTest(unittest.TestCase):
    def test_duration_device_and_future_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def make(name, device, seconds, wall, stamp="2026-01-01T00:00:00+00:00"):
                run = root / name
                (run / "work").mkdir(parents=True)
                (run / "work/media-source.json").write_text(json.dumps({"device": device, "duration_s": seconds}))
                (run / "work/progress.json").write_text(json.dumps({"status": "completed", "updated_at": stamp}))
                (run / "performance.json").write_text(json.dumps({"stages": [{"name": "stage.speaker", "wall_seconds": wall}]}))
                return run
            make("cpu", "cpu", 100, 200)
            make("gpu", "cuda:0", 100, 10)
            make("future", "cpu", 100, 9000, "2028-01-01T00:00:00+00:00")
            current = make("current", "cpu", 10, 999)
            started = ps.parse_time("2027-01-01T00:00:00+00:00")
            short, info = ps.historical_baseline(current, started)
            long, _ = ps.historical_baseline(current, started, duration=1000)
            self.assertEqual(info["history_runs"], 1)
            self.assertEqual(short["speaker"], 20)
            self.assertEqual(long["speaker"], 2000)
            gpu, _ = ps.historical_baseline(current, started, device="gpu")
            self.assertEqual(gpu["speaker"], 1)
            unknown, meta = ps.historical_baseline(current, started, device="unknown")
            self.assertEqual(meta["history_runs"], 0)

    def test_uncertainty_and_terminal_states(self):
        exp = dict(ps.DEFAULT_SECONDS)
        meta = {"duration_s": 100, "basis": "test"}
        for done in (0, 1):
            snap = ps.build_snapshot(progress(done), exp, T0, T0 + 10, estimate=meta)
            self.assertIsNone(snap["overall"]["remaining_s"])
        stamp = ps.parse_time("2026-01-01T00:00:00+00:00")
        snap = ps.build_snapshot(progress(4), exp, stamp - 100, stamp + 5000, estimate=meta)
        self.assertIsNone(snap["overall"]["remaining_s"])
        snap = ps.build_snapshot(progress(4), exp, stamp - 100, stamp + 1, estimate={"duration_s": None})
        self.assertIsNone(snap["overall"]["eta_unix_s"])
        for state in ("failed", "cancelled"):
            snap = ps.build_snapshot(progress(12, "completed"), exp, stamp - 100, stamp + 1, state, meta)
            self.assertEqual(snap["run_status"], state)
            self.assertIsNone(snap["overall"]["remaining_s"])
        a = ps.build_snapshot(progress(12, "completed"), exp, stamp - 100, stamp + 100, estimate=meta)
        b = ps.build_snapshot(progress(12, "completed"), exp, stamp - 100, stamp + 1000, estimate=meta)
        self.assertEqual(a["overall"]["elapsed_s"], b["overall"]["elapsed_s"])

    def test_explicit_completed_state_freezes_clock(self):
        stamp = ps.parse_time('2026-01-01T00:00:00+00:00')
        results = [ps.build_snapshot(progress(12, 'completed'), dict(ps.DEFAULT_SECONDS),
                                     stamp - 100, stamp + drift, 'completed') for drift in (100, 1000)]
        self.assertEqual([s['overall']['elapsed_s'] for s in results], [100, 100])

    def test_gpu_without_history_has_no_countdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected, meta = ps.historical_baseline(Path(tmp) / 'current', T0, duration=100, device='gpu')
            stamp = ps.parse_time('2026-01-01T00:00:00+00:00')
            snap = ps.build_snapshot(progress(4), expected, stamp - 50, stamp + 1, estimate=meta)
            self.assertIsNone(snap['overall']['remaining_s'])

    def test_cloud_cache_labels_do_not_scale_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'current'
            (run / 'work').mkdir(parents=True)
            expected = []
            for requests, hits, label in ((0, 3, 'warm'), (2, 1, 'cold_or_mixed')):
                (run / 'work/cloud-asr-usage.json').write_text(json.dumps(
                    {'status': 'completed', 'requests': requests, 'cache_hits': hits}))
                values, meta = ps.historical_baseline(run, T0, duration=100, device='cpu')
                self.assertEqual(meta['asr_cache'], label)
                expected.append(values['speaker'])
            self.assertEqual(expected[0], expected[1])



if __name__ == "__main__":
    unittest.main()

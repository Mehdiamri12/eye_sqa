"""
test_pipeline.py — Signal integrity and parser tests.

Run all:
    pytest tests/test_pipeline.py -v -s

Run one class:
    pytest tests/test_pipeline.py::TestParserRobustness -v -s
    pytest tests/test_pipeline.py::TestCTLESweep -v -s

Run one test:
    pytest tests/test_pipeline.py::TestCTLESweep::test_sweep_finds_correct_optimal_boost -v -s

Break a test deliberately (see docstrings for instructions):
    Edit pipeline.py → run → observe the failure → restore pipeline.py

Ground truth (measured from real SPICE data):
    PAM3 80ui:  baseline=174.9mV  optimal=2.0dB→204.2mV  cliff at 4.5dB→7.3mV
    PAM2 50ui:  baseline=364.6mV  optimal=3.5dB→443.0mV  plateau stays >400mV
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_spice_txt
from interpolate import interpolate_waveform
from pipeline import get_pam_config, apply_ctle, run_fold_and_measure
from tests.conftest import SIGNALS, SPICE_AVAILABLE

BOOST_SWEEP = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]


def run_sweep(t, v, ui, path):
    """Run full CTLE sweep and return {db: eye_height_mv} dict."""
    results = {}
    for db in BOOST_SWEEP:
        v_eq, _, _, _, _ = apply_ctle(t, v.copy(), ui, db)
        _, lv, rv         = get_pam_config(v_eq, path)
        eh                = run_fold_and_measure(t, v_eq, ui, rv, lv)[-2]
        results[db]       = eh
        print(f"    {db:>4}dB → EH={eh*1000:.1f}mV")
    return results


skip_no_data = pytest.mark.skipif(
    not SPICE_AVAILABLE,
    reason="SPICE data files not found in Data/ — copy your Data/ folder to run these tests"
)


# ════════════════════════════════════════════════════════════════════════════
# TestParserRobustness
#
# The parser is the entry point of the entire pipeline.
# A silent parser failure (wrong length, NaN, non-monotonic time)
# corrupts every downstream measurement without any visible error.
# ════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestParserRobustness:
    """
    Tests the parser on real SPICE files (tab-separated, scientific notation)
    and on synthetically corrupted files created with tmp_path.

    HOW TO BREAK these tests:
        test_time_axis_is_strictly_monotonic:
            In parser.py, reverse the time sort or skip the sort step.
            → diffs will contain negative values → assert fails.

        test_voltage_values_are_finite:
            In parser.py, replace float(parts[1]) with float('nan')
            for the first row. → np.isnan check fails.

        test_parser_raises_on_corrupted_file:
            Remove the try/except in parser.py or return empty arrays
            instead of raising. → pytest.raises block fails.
    """

    @pytest.mark.parametrize("key", ["pam3_80ui", "pam2_50ui"])
    def test_time_axis_is_strictly_monotonic(self, key):
        """
        Time must be strictly increasing — no duplicates, no backwards steps.

        Non-monotonic time silently corrupts phase folding:
        (t - offset) % ui produces wrong phase values, the eye diagram
        becomes a smeared blob, and metrics are meaningless.

        Parametrized across both PAM3 and PAM2 files.
        """
        cfg    = SIGNALS[key]
        path   = cfg["path"]
        print(f"\n[parser] Loading {os.path.basename(path)} ...")

        t, v = parse_spice_txt(path)
        print(f"[parser] {len(t):,} samples parsed")

        diffs    = np.diff(t)
        n_bad    = np.sum(diffs <= 0)
        min_step = diffs.min() * 1e12

        print(f"[parser] Time axis: min_step={min_step:.4f}ps  non-monotonic={n_bad}")

        assert n_bad == 0, (
            f"{os.path.basename(path)}: {n_bad} non-increasing time steps. "
            f"Phase folding will produce a corrupted eye diagram."
        )

    @pytest.mark.parametrize("key", ["pam3_80ui", "pam2_50ui"])
    def test_voltage_values_are_finite(self, key):
        """
        Voltage array must contain no NaN or Inf.

        A single NaN propagates silently through numpy:
            np.mean([1.0, NaN, 2.0]) = NaN
            → CTLE DC removal fails → entire signal becomes NaN
            → eye diagram collapses to zero without any error message.

        Inf causes get_pam_config to compute v_min=Inf, v_max=Inf
        → np.linspace(Inf, Inf, 3) = [Inf, Inf, Inf]
        → all decision thresholds are Inf → eye height = 0.

        Both are silent failures — pipeline runs to completion,
        returns wrong metrics, engineer makes a bad design decision.
        """
        cfg  = SIGNALS[key]
        path = cfg["path"]
        print(f"\n[parser] Loading {os.path.basename(path)} ...")

        _, v  = parse_spice_txt(path)
        n_nan = int(np.sum(np.isnan(v)))
        n_inf = int(np.sum(np.isinf(v)))

        print(f"[parser] Voltage: range=[{v.min():.4f}, {v.max():.4f}]V  "
              f"NaN={n_nan}  Inf={n_inf}")

        assert n_nan == 0, (
            f"{os.path.basename(path)}: {n_nan} NaN values. "
            f"NaN propagates silently through CTLE — corrupts all metrics."
        )
        assert n_inf == 0, (
            f"{os.path.basename(path)}: {n_inf} Inf values. "
            f"Inf collapses PAM level detection."
        )

    def test_parser_raises_on_corrupted_file(self, tmp_path):
        """
        A single-column file (missing voltage — truncated export)
        must raise an exception, not return empty or partial arrays.

        Without this guard, a truncated SPICE file produces
        a zero-length voltage array, and run_fold_and_measure
        crashes with an uninformative IndexError deep in the pipeline.
        Better to fail loudly at the entry point.

        HOW TO BREAK: change parse_spice_txt to return ([], [])
        instead of raising → pytest.raises block fails.
        """
        bad_file = tmp_path / "truncated.txt"
        bad_file.write_text(
            "time\tV(rx_input)\n"
            "0.000000000000000e+00\n"         # missing voltage column
            "9.089660644531248e-13\n"
        )
        print(f"\n[parser] Testing truncated file: {bad_file}")

        with pytest.raises(Exception) as exc_info:
            parse_spice_txt(str(bad_file))

        print(f"[parser] Correctly raised: {type(exc_info.value).__name__}: {exc_info.value}")


# ════════════════════════════════════════════════════════════════════════════
# TestCTLESweep
#
# The CTLE sweep finds the boost level that maximises worst-case eye height.
# An engineer uses this output to select the production CTLE setting.
# If the sweep returns the wrong optimal, the chip ships with suboptimal
# signal margin — a silent failure with real hardware consequences.
# ════════════════════════════════════════════════════════════════════════════

@skip_no_data
class TestCTLESweep:
    """
    Tests the CTLE sweep on both channels with very different equalization profiles:

    PAM3 80ui — sharp optimum at 2 dB, catastrophic collapse above 4.5 dB.
                Over-equalization is dangerous on this channel.

    PAM2 50ui — broad plateau peaking at 3.5 dB, stays >400mV from 2-4.5 dB.
                Much more forgiving — wider production tolerance.

    HOW TO BREAK these tests:

        test_sweep_finds_correct_optimal_boost:
            In pipeline.py → apply_ctle → change:
                boost = 10 ** (boost_db / 20)
            to:
                boost = 10 ** (boost_db / 10)   # wrong: power dB instead of amplitude dB
            Effect: every boost applies twice as much equalization.
            The sweep curve shifts left — PAM3 optimal moves from 2dB to 1dB,
            PAM2 optimal moves from 3.5dB to 1.5dB.
            Full sweep printed in error message so you see the shifted curve.

        test_pam3_cliff_is_detected:
            In pipeline.py → get_pam_config → change:
                elif "pam3" in filename: pam_order = 3
            to:
                elif "pam3x" in filename: pam_order = 3
            Effect: PAM3 file processed as PAM2 — wrong thresholds,
            all eye heights collapse to ~0mV. Sweep reports 4.5dB as
            "optimal" with 0.5mV because it's the least-wrong value.
            This is the silent wrong answer — most dangerous bug.

        test_optimal_boost_improves_baseline:
            In pipeline.py → apply_ctle → change:
                H = 1 + (boost - 1) * 4 * S
            to:
                H = 1 - (boost - 1) * 4 * S   # inverted filter
            Effect: CTLE now attenuates high frequencies instead of boosting.
            At 2dB: EH drops from 174.9mV to ~167mV → improvement = -7.9mV.
            Fails with: "CTLE degraded the eye by -7.9mV instead of improving it."
    """

    @pytest.mark.parametrize("key", ["pam3_80ui", "pam2_50ui"])
    def test_sweep_finds_correct_optimal_boost(self, key, pam3, pam2):
        """
        The CTLE sweep must identify the same optimal boost as the
        pre-measured ground truth on real SPICE data.

        PAM3 optimal: 2.0 dB → 204.2 mV
        PAM2 optimal: 3.5 dB → 443.0 mV

        Golden master test. If the sweep returns a different optimal,
        either the dB formula, the transfer function, or the metric
        computation has changed. The full sweep is printed so you can
        see exactly where the curve shifted.
        """
        cfg     = SIGNALS[key]
        t, v, ui, _, _ = pam3 if key == "pam3_80ui" else pam2

        print(f"\n[sweep] Running CTLE sweep on {os.path.basename(cfg['path'])} ...")
        results = run_sweep(t, v, ui, cfg["path"])

        best_db  = max(results, key=results.get)
        best_eh  = results[best_db] * 1000
        sweep_str = "  |  ".join(f"{d}dB={e*1000:.0f}mV" for d, e in results.items())

        print(f"[sweep] Best: {best_db}dB → {best_eh:.1f}mV  "
              f"(expected {cfg['optimal_db']}dB → {cfg['optimal_eh']:.1f}mV)")

        assert best_db == cfg["optimal_db"], (
            f"{os.path.basename(cfg['path'])}: "
            f"Sweep found optimal at {best_db}dB ({best_eh:.1f}mV) "
            f"but expected {cfg['optimal_db']}dB ({cfg['optimal_eh']:.1f}mV).\n"
            f"Full sweep: {sweep_str}"
        )

    @pytest.mark.parametrize("key", ["pam3_80ui", "pam2_50ui"])
    def test_optimal_boost_improves_baseline(self, key, pam3, pam2):
        """
        Applying the measured optimal boost must improve eye height
        over baseline by at least the expected margin.

        PAM3: +29.3 mV  (174.9 → 204.2 mV)
        PAM2: +78.4 mV  (364.6 → 443.0 mV)

        If CTLE makes things worse, the filter is inverted or broken.
        """
        cfg     = SIGNALS[key]
        t, v, ui, lv, rv = pam3 if key == "pam3_80ui" else pam2

        print(f"\n[sweep] Checking CTLE improvement on {os.path.basename(cfg['path'])} ...")

        eh_base = run_fold_and_measure(t, v.copy(), ui, rv, lv)[-2]

        v_opt, _, _, _, _ = apply_ctle(t, v.copy(), ui, cfg["optimal_db"])
        _, lv_opt, rv_opt = get_pam_config(v_opt, cfg["path"])
        eh_opt            = run_fold_and_measure(t, v_opt, ui, rv_opt, lv_opt)[-2]
        gain_mv           = (eh_opt - eh_base) * 1000

        print(f"[sweep] Baseline={eh_base*1000:.1f}mV  "
              f"After {cfg['optimal_db']}dB CTLE={eh_opt*1000:.1f}mV  "
              f"Gain={gain_mv:+.1f}mV  (min required: {cfg['min_gain_mv']:.0f}mV)")

        assert gain_mv >= cfg["min_gain_mv"], (
            f"{os.path.basename(cfg['path'])}: "
            f"CTLE at {cfg['optimal_db']}dB degraded or barely improved the eye. "
            f"Gain={gain_mv:.1f}mV, required >={cfg['min_gain_mv']:.0f}mV. "
            f"Baseline={eh_base*1000:.1f}mV → After CTLE={eh_opt*1000:.1f}mV."
        )

    def test_pam3_cliff_is_detected(self, pam3):
        """
        PAM3 80ui must show catastrophic eye closure above 4.5 dB boost.

        Measured: 4.5 dB → 7.3 mV (from 204.2 mV optimal — 96% collapse).

        This test is PAM3-specific because PAM2 50ui stays healthy
        at the same boost level (434 mV at 5 dB). The two channels
        have fundamentally different equalization tolerances.

        An engineer selecting the CTLE boost for PAM3 must know this cliff
        exists. If the pipeline reports 4.5 dB as safe on PAM3, they
        ship a chip with a near-closed eye.

        HOW TO BREAK: in pipeline.py change 'pam3x' so PAM3 files are
        processed as PAM2 — all sweep values collapse to ~0mV, cliff
        is no longer detectable.
        """
        cfg = SIGNALS["pam3_80ui"]
        t, v, ui, _, _ = pam3

        print(f"\n[cliff] Testing PAM3 over-equalization cliff ...")
        v_over, _, _, _, _ = apply_ctle(t, v.copy(), ui, cfg["overboost_db"])
        _, lv_o, rv_o      = get_pam_config(v_over, cfg["path"])
        eh_cliff           = run_fold_and_measure(t, v_over, ui, rv_o, lv_o)[-2]

        DANGER_THRESHOLD = 0.020   # 20 mV — well below any healthy eye

        print(f"[cliff] At {cfg['overboost_db']}dB: EH={eh_cliff*1000:.1f}mV  "
              f"(threshold={DANGER_THRESHOLD*1000:.0f}mV  "
              f"measured ground truth={cfg['overboost_eh']:.1f}mV)")

        assert eh_cliff < DANGER_THRESHOLD, (
            f"PAM3 80ui at {cfg['overboost_db']}dB: "
            f"EH={eh_cliff*1000:.1f}mV — expected collapse below {DANGER_THRESHOLD*1000:.0f}mV. "
            f"Measured ground truth: {cfg['overboost_eh']:.1f}mV. "
            f"The over-equalization cliff is not being captured correctly."
        )

    def test_pam2_plateau_remains_safe_across_boost_range(self, pam2):
        """
        PAM2 50ui must maintain eye height above 400 mV across 2-4.5 dB.

        Measured: all values in [415, 443] mV across this range.
        This documents the broad plateau — useful for production tolerance
        analysis. How tight does the CTLE boost selection need to be?
        For PAM2 50ui: loose. For PAM3 80ui: very tight.

        PAM3 would fail this test (collapses at 4.5 dB) — that's why
        this is PAM2-specific. The contrast between the two channels
        is the key engineering insight.
        """
        cfg = SIGNALS["pam2_50ui"]
        t, v, ui, _, _ = pam2

        PLATEAU_RANGE  = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        HEALTHY_FLOOR  = 0.400   # 400 mV

        print(f"\n[plateau] PAM2 50ui plateau test across {PLATEAU_RANGE}dB ...")
        for db in PLATEAU_RANGE:
            v_eq, _, _, _, _ = apply_ctle(t, v.copy(), ui, db)
            _, lv_eq, rv_eq  = get_pam_config(v_eq, cfg["path"])
            eh               = run_fold_and_measure(t, v_eq, ui, rv_eq, lv_eq)[-2]
            status           = "OK" if eh >= HEALTHY_FLOOR else "FAIL"
            print(f"  [{status}] {db}dB → EH={eh*1000:.1f}mV  "
                  f"(floor={HEALTHY_FLOOR*1000:.0f}mV)")
            assert eh >= HEALTHY_FLOOR, (
                f"PAM2 50ui at {db}dB: EH={eh*1000:.1f}mV dropped below "
                f"{HEALTHY_FLOOR*1000:.0f}mV plateau floor. "
                f"The broad-plateau characteristic has changed."
            )

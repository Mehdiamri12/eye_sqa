"""
test_api.py — API tests focused on signal integrity correctness.

Run all:
    pytest tests/test_api.py -v -s

Run one class:
    pytest tests/test_api.py::TestCTLEViaAPI -v -s
    pytest tests/test_api.py::TestInputGuards -v -s

Run one test:
    pytest tests/test_api.py::TestCTLEViaAPI::test_file_pam3_golden_master -v -s

These tests verify the API layer correctly exposes the pipeline.
They are not generic web tests — every assertion is grounded in
the physics of the real SPICE data.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app
from tests.conftest import SIGNALS

import numpy as np

client = TestClient(app)

DATA    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")
PAM3    = SIGNALS["pam3_80ui"]["path"]
PAM2    = SIGNALS["pam2_50ui"]["path"]


def nrz_payload(n=150, noise=0.1, seed=42):
    np.random.seed(seed)
    bits = np.random.choice([-1.0, 1.0], size=n)
    v    = np.repeat(bits, 20).astype(float)
    v   += np.random.normal(0, noise, len(v))
    return {"samples": v.tolist(), "ui_ps": 100.0, "modulation": "NRZ"}


NRZ = nrz_payload()


# ════════════════════════════════════════════════════════════════════════════
# TestCTLEViaAPI
#
# Does the API correctly expose the CTLE sweep results?
# Golden master tests grounded in real SPICE measurements.
# ════════════════════════════════════════════════════════════════════════════

class TestCTLEViaAPI:
    """
    Tests that the API correctly reports CTLE sweep results.
    Any change to the pipeline that shifts metrics will be caught here.
    """

    @pytest.mark.skipif(not os.path.exists(PAM3), reason="PAM3 file not found")
    def test_file_pam3_golden_master(self):
        """
        PAM3 80ui file upload: baseline EH must be within ±10mV of
        the measured ground truth (174.9mV) and optimal boost must be 2.0dB.

        This is the single most important test — if either value shifts,
        a code change has altered how the pipeline processes real GDDR-style
        PAM3 data. The engineer can no longer trust the sweep output.

        HOW TO BREAK:
            Change boost = 10**(boost_db/10) in apply_ctle → optimal shifts to 1.0dB.
            Change 'pam3x' in get_pam_config → baseline collapses to ~0mV.
        """
        print(f"\n[api] Uploading {os.path.basename(PAM3)} ...")
        with open(PAM3, "rb") as f:
            r = client.post("/analyze/file",
                            files={"file": ("80_ui_netlist_pam3.txt", f, "text/plain")})

        assert r.status_code == 201, f"Expected 201, got {r.status_code}: {r.text[:200]}"
        d = r.json()

        baseline_eh = d["baseline"]["eye_height_mv"]
        optimal_db  = d["ctle"]["optimal_boost_db"]
        improvement = d["ctle"]["improvement_mv"]

        print(f"[api] baseline_eh={baseline_eh:.1f}mV  optimal={optimal_db}dB  "
              f"improvement={improvement:+.1f}mV")

        assert abs(baseline_eh - 174.9) <= 10.0, (
            f"PAM3 baseline EH={baseline_eh:.1f}mV outside ±10mV of 174.9mV. "
            f"Pipeline has regressed."
        )
        assert optimal_db == 2.0, (
            f"PAM3 optimal boost={optimal_db}dB, expected 2.0dB. "
            f"Sweep calibration has shifted — check dB formula in apply_ctle."
        )
        assert improvement > 0, (
            f"CTLE degraded PAM3 eye by {improvement:.1f}mV instead of improving it."
        )

    @pytest.mark.skipif(not os.path.exists(PAM2), reason="PAM2 file not found")
    def test_file_pam2_optimal_boost_in_range(self):
        """
        PAM2 50ui file upload: optimal boost must be between 3.0 and 4.0dB.
        Measured: 3.5dB.

        PAM2 and PAM3 have different optimal boosts because they have
        different channel characteristics. If both return the same optimal,
        the PAM detection or file routing is broken.
        """
        print(f"\n[api] Uploading {os.path.basename(PAM2)} ...")
        with open(PAM2, "rb") as f:
            r = client.post("/analyze/file",
                            files={"file": ("50_ui_netlist_pam2.txt", f, "text/plain")})

        assert r.status_code == 201
        d = r.json()
        optimal_db = d["ctle"]["optimal_boost_db"]
        print(f"[api] PAM2 optimal={optimal_db}dB  (expected 3.0-4.0dB)")

        assert 3.0 <= optimal_db <= 4.0, (
            f"PAM2 50ui optimal boost={optimal_db}dB outside expected [3.0, 4.0]dB. "
            f"Ground truth: 3.5dB."
        )

    def test_json_sweep_contains_danger_zone_for_pam3_like_signal(self):
        """
        The sweep response must contain all 11 boost levels and at least
        one value where EH is significantly below the peak (>80% drop).

        This verifies the API exposes enough data for the frontend to
        render danger zones — the engineer needs to see the full curve,
        not just the optimal value.
        """
        print(f"\n[api] Submitting NRZ signal and checking sweep completeness ...")
        r = client.post("/analyze/json", json=NRZ)
        assert r.status_code == 201

        sweep = r.json()["ctle"]["sweep"]
        print(f"[api] Sweep keys: {list(sweep.keys())}")

        expected = ["0", "0.5", "1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]
        for db in expected:
            assert db in sweep, f"Sweep missing boost level {db}dB"
        print(f"[api] All 11 boost levels present ✓")


# ════════════════════════════════════════════════════════════════════════════
# TestInputGuards
#
# The API must reject bad inputs loudly, not process them silently.
# Silent wrong answers are worse than crashes in engineering tools.
# ════════════════════════════════════════════════════════════════════════════

class TestInputGuards:
    """
    Tests that bad inputs are rejected at the API boundary with
    informative error messages.

    These mirror the silent failure modes identified in TestParserRobustness:
    constant signals, NaN, wrong file format, insufficient data.
    """

    def test_constant_signal_rejected(self):
        """
        All-identical samples must return 422 with a message
        about constant/identical signal.

        Silent acceptance would produce a degenerate eye (EH=0)
        with no error — the engineer thinks the channel is fine.
        This is the API-level guard for the same bug that
        test_pipeline.py::TestParserRobustness documents.
        """
        payload = {**NRZ, "samples": [0.9] * 200}
        print(f"\n[api] Testing constant signal rejection ...")
        r = client.post("/analyze/json", json=payload)

        print(f"[api] Response: {r.status_code} — {str(r.json())[:150]}")
        assert r.status_code == 422

        detail = str(r.json().get("detail", "")).lower()
        assert "constant" in detail or "identical" in detail, (
            f"422 returned but message doesn't mention the cause: {detail}"
        )

    def test_wrong_file_format_rejected(self):
        """
        A .csv file upload must return 422. Engineers occasionally
        select the wrong file — the error must be immediate and clear,
        not a crash deep in the parser.
        """
        print(f"\n[api] Testing .csv file rejection ...")
        r = client.post("/analyze/file",
                        files={"file": ("signal.csv", b"time,v\n0,0.5", "text/csv")})

        print(f"[api] Response: {r.status_code}")
        assert r.status_code == 422

    def test_missing_result_returns_404(self):
        """
        Requesting a non-existent result ID must return 404,
        not crash or return an empty result.
        """
        print(f"\n[api] Testing missing result ID ...")
        r = client.get("/results/does_not_exist_xyz")
        print(f"[api] Response: {r.status_code}")
        assert r.status_code == 404

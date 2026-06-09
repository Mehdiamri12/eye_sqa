"""
test_e2e.py — End-to-end browser tests focused on engineering-relevant UI behaviour.

Run all:
    pytest tests/test_e2e.py -v -s

Run one test:
    pytest tests/test_e2e.py::TestFrontendFlow::test_pam3_cliff_shown_in_red -v -s
    pytest tests/test_e2e.py::TestFrontendFlow::test_constant_signal_shows_error -v -s
    pytest tests/test_e2e.py::TestFrontendFlow::test_pam3_file_upload_full_flow -v -s

Requires:
    pip install playwright uvicorn
    playwright install chromium

The browser stays open after each test (page.pause() in conftest.py).
Press Resume in the Playwright Inspector toolbar to continue to the next test.

These tests verify that the frontend correctly communicates signal quality
to the engineer — not just that the API returns correct JSON.
"""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.conftest import SIGNALS

BASE_URL = "http://localhost:8000"
PAM3_FILE = SIGNALS["pam3_80ui"]["path"]


def nrz_samples_str(n=150, noise=0.1, seed=42):
    """Comma-separated NRZ samples for the textarea."""
    np.random.seed(seed)
    bits = np.random.choice([-1.0, 1.0], size=n)
    v    = np.repeat(bits, 20).astype(float)
    v   += np.random.normal(0, noise, len(v))
    return ", ".join(f"{s:.4f}" for s in v)


class TestFrontendFlow:
    """
    Three focused E2E tests — each demonstrates a key engineering behaviour:

    1. PAM3 over-equalization cliff → red danger cells in sweep grid
    2. Constant signal → error card, no result rendered
    3. Real PAM3 file upload → full result with sweep, stage bars, metrics
    """

    @pytest.mark.skipif(not os.path.exists(PAM3_FILE), reason="PAM3 SPICE file not available")
    def test_pam3_cliff_shown_in_red(self, page):
        """
        Upload the real PAM3 80ui SPICE file and verify that the
        over-equalization cliff (4.5-5.0 dB → <4% of peak EH) is
        rendered as red danger cells in the sweep grid.

        This is the most engineering-relevant visual test:
        - Green cell at 2 dB = safe operating point
        - Red cells at 4.5-5 dB = dangerous — near-closed eye
        The engineer must see this without reading numbers.

        What you should see in the browser:
            The sweep grid appears with one green cell (2dB = 204mV)
            and two red cells (4.5dB = 7mV, 5dB = 1mV).
        """
        if not os.path.exists(PAM3_FILE):
            pytest.skip("PAM3 SPICE file not available")

        print(f"\n[e2e] Uploading {os.path.basename(PAM3_FILE)} ...")
        page.goto(BASE_URL)
        page.locator("#spice-file").set_input_files(PAM3_FILE)
        page.click("#upload-btn")

        print("[e2e] Waiting for result ...")
        page.locator("#result-section").wait_for(state="visible", timeout=90000)

        danger_cells = page.locator(".sweep-cell.danger").all()
        best_cells   = page.locator(".sweep-cell.best").all()

        print(f"[e2e] Sweep grid: {len(best_cells)} green (optimal), "
              f"{len(danger_cells)} red (danger)")

        assert len(best_cells) == 1, (
            f"Expected 1 green optimal cell, got {len(best_cells)}. "
            f"Optimal boost (2dB) not highlighted."
        )
        assert len(danger_cells) >= 1, (
            f"Expected ≥1 red danger cell, got {len(danger_cells)}. "
            f"PAM3 over-equalization cliff at 4.5-5dB not shown. "
            f"Threshold: cells with <20% of peak EH rendered red."
        )

        baseline_eh = page.locator("#m-base-eh").inner_text()
        print(f"[e2e] Baseline EH displayed: {baseline_eh}mV")
        assert float(baseline_eh) > 0

    def test_constant_signal_shows_error(self, page):
        """
        Submit 200 identical samples — the error card must appear
        and the result section must stay hidden.

        This is the E2E version of the silent failure regression test.
        The API returns 422. The frontend must show a clear error,
        not a blank result or a zero eye height that looks like a real measurement.

        What you should see in the browser:
            The error card appears with a red title and description.
            The result section (metrics, sweep grid, stage bars) stays hidden.
        """
        print(f"\n[e2e] Testing constant signal error display ...")
        page.goto(BASE_URL)
        page.select_option("#modulation", "NRZ")
        page.fill("#ui-ps", "100")
        page.fill("#samples-input", ", ".join(["0.9"] * 200))
        page.click("#analyze-btn")

        error = page.locator("#error-section")
        error.wait_for(state="visible", timeout=10000)

        result = page.locator("#result-section")
        print(f"[e2e] error_visible={error.is_visible()}  result_visible={result.is_visible()}")

        assert error.is_visible(), (
            "Error section not shown for constant signal. "
            "Engineer would see no feedback."
        )
        assert not result.is_visible(), (
            "Result section visible despite error. "
            "Engineer could mistake the error state for a valid zero-EH result."
        )

        error_text = page.locator("#error-text").inner_text()
        print(f"[e2e] Error message: '{error_text}'")
        assert len(error_text) > 0, "Error text is empty — no message shown to engineer."

    @pytest.mark.skipif(not os.path.exists(PAM3_FILE), reason="PAM3 SPICE file not available")
    def test_pam3_file_upload_full_flow(self, page):
        """
        Full end-to-end: upload PAM3 SPICE file → verify metrics render,
        sweep grid shows 11 cells, stage bars show all 3 pipeline stages.

        This test covers the complete engineer workflow:
        select file → click analyze → read results.

        What you should see in the browser:
            Stage bars showing Baseline → CTLE → CTLE+DFE progression.
            Sweep grid with 11 cells, one green at 2dB.
            Baseline EH ~175mV, CTLE EH ~204mV.
        """
        if not os.path.exists(PAM3_FILE):
            pytest.skip("PAM3 SPICE file not available")

        print(f"\n[e2e] Full flow test — uploading {os.path.basename(PAM3_FILE)} ...")
        page.goto(BASE_URL)
        page.locator("#spice-file").set_input_files(PAM3_FILE)
        page.click("#upload-btn")

        page.locator("#result-section").wait_for(state="visible", timeout=90000)

        # Verify all result components rendered
        sweep_cells = page.locator(".sweep-cell").all()
        stage_rows  = page.locator(".stage-row").all()
        baseline_eh = page.locator("#m-base-eh").inner_text()
        optimal_db  = page.locator("#m-boost").inner_text()

        print(f"[e2e] sweep_cells={len(sweep_cells)}  stage_rows={len(stage_rows)}")
        print(f"[e2e] baseline_eh={baseline_eh}mV  optimal_boost={optimal_db}dB")

        assert len(sweep_cells) == 11, (
            f"Expected 11 sweep cells (0-5dB), got {len(sweep_cells)}."
        )
        assert len(stage_rows) == 3, (
            f"Expected 3 stage bars (Baseline, CTLE, CTLE+DFE), got {len(stage_rows)}."
        )
        assert baseline_eh != "—" and float(baseline_eh) > 0, (
            f"Baseline EH not rendered: '{baseline_eh}'"
        )
        assert optimal_db != "—", "Optimal boost not rendered."

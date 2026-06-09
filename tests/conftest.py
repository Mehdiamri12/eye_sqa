"""
tests/conftest.py — Shared fixtures for unit and API tests.

Loads the two SPICE files once per test session and caches the results.
This avoids re-parsing 138k-point files on every test.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser import parse_spice_txt
from interpolate import interpolate_waveform
from pipeline import get_pam_config, run_fold_and_measure

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data")

# Skip the entire pipeline + API test session if SPICE files are not present.
# This allows CI to run E2E tests independently without needing proprietary data.
SPICE_AVAILABLE = (
    os.path.exists(os.path.join(DATA, "PAM3", "80_ui_netlist_pam3.txt")) and
    os.path.exists(os.path.join(DATA, "PAM2", "50_ui_netlist_pam2.txt"))
)

if not SPICE_AVAILABLE:
    import warnings
    warnings.warn(
        "\n[conftest] SPICE data files not found in Data/. "
        "Pipeline and API tests will be skipped. "
        "Copy your Data/ folder into the project to run them.",
        UserWarning
    )

SIGNALS = {
    "pam3_80ui": {
        "path":        os.path.join(DATA, "PAM3", "80_ui_netlist_pam3.txt"),
        "baud":        1 / 80 * 1e12,
        "pam_order":   3,
        "baseline_eh": 174.9,   # mV — measured from real SPICE data
        "optimal_db":  2.0,
        "optimal_eh":  204.2,   # mV
        "overboost_db":  4.5,   # dB — where PAM3 collapses catastrophically
        "overboost_eh":  7.3,   # mV — 96% collapse from optimal
        "min_gain_mv":  20.0,   # minimum acceptable CTLE improvement
    },
    "pam2_50ui": {
        "path":        os.path.join(DATA, "PAM2", "50_ui_netlist_pam2.txt"),
        "baud":        1 / 50 * 1e12,
        "pam_order":   2,
        "baseline_eh": 364.6,   # mV
        "optimal_db":  3.5,
        "optimal_eh":  443.0,   # mV
        "overboost_db":  5.0,   # dB — past plateau, degrading
        "overboost_eh":  434.1, # mV — gentle degradation, not a cliff
        "min_gain_mv":  50.0,
    },
}


def load_signal(cfg):
    """Parse + interpolate one SPICE file. Returns (t, v, ui, v_levels, v_refs)."""
    print(f"\n  [fixture] Loading {os.path.basename(cfg['path'])} ...")
    t, v, ui = interpolate_waveform(*parse_spice_txt(cfg["path"]), cfg["baud"])
    _, v_levels, v_refs = get_pam_config(v, cfg["path"])
    print(f"  [fixture] {len(t):,} pts | UI={ui*1e12:.0f}ps | "
          f"PAM-{len(v_levels)} | levels={[round(l,3) for l in v_levels]}")
    return t, v, ui, v_levels, v_refs


@pytest.fixture(scope="session")
def pam3(request):
    return load_signal(SIGNALS["pam3_80ui"])


@pytest.fixture(scope="session")
def pam2(request):
    return load_signal(SIGNALS["pam2_50ui"])

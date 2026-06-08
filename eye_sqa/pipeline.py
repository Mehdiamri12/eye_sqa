"""
pipeline.py — Pure signal processing functions for the eye diagram pipeline.

Extracted from test.py (the original CTLE+DFE pipeline).
No file I/O, no plotting, no settings dependency — safe to import anywhere.

Functions:
    get_pam_config       — detect PAM order from filename, compute levels/refs
    apply_ctle           — frequency-domain CTLE equalizer
    get_phase_offset     — derivative-weighted histogram clock recovery
    apply_dfe            — symbol-by-symbol decision feedback equalizer
    run_fold_and_measure — fold waveform onto one UI, compute eye metrics

Copy these files from your original project into this folder:
    parser.py, interpolate.py, metrics.py, settings.py, test.py
"""

import numpy as np
from metrics import measure_eye_metrics


def get_pam_config(v_signal, filepath):
    filename = str(filepath).lower()
    if   "pam4" in filename: pam_order = 4
    elif "pam3" in filename: pam_order = 3
    else:                     pam_order = 2

    v_min    = np.min(v_signal)
    v_max    = np.max(v_signal)
    v_levels = list(np.linspace(v_min, v_max, pam_order))
    v_refs   = [(v_levels[i] + v_levels[i+1]) / 2.0
                for i in range(pam_order - 1)]
    return pam_order, v_levels, v_refs


def apply_ctle(t, v, ui, boost_db):
    dt      = np.median(np.diff(t))
    dc      = np.mean(v)
    v_ac    = v - dc
    V       = np.fft.rfft(v_ac)
    f       = np.fft.rfftfreq(len(v), d=dt)
    f_nyq   = 1 / (2 * ui)
    boost   = 10 ** (boost_db / 20)
    f_ratio = f / (f_nyq + 1e-30)
    S       = f_ratio**2 / (1 + f_ratio**2)**2
    H       = 1 + (boost - 1) * 4 * S
    v_eq    = np.fft.irfft(V * H, n=len(v)) + dc
    return v_eq, f, V, np.fft.rfft(v_eq - dc), H


def get_phase_offset(t_uni, v, ui):
    dv          = np.abs(np.diff(v))
    t_folded    = t_uni[:-1] % ui
    hist, edges = np.histogram(t_folded, bins=100, weights=dv)
    peak        = np.argmax(hist)
    return (edges[peak] + edges[peak + 1]) / 2.0


def apply_dfe(t, v, ui, levels, offset, n_taps=1, tap_weight=0.05):
    dt            = np.median(np.diff(t))
    corrected     = v.copy()
    levels_sorted = np.sort(levels)
    mid_level     = (levels_sorted[0] + levels_sorted[-1]) / 2.0
    t_start       = t[0]
    remainder     = t_start % ui
    first_crossing = (t_start - remainder + offset
                      if remainder <= offset
                      else t_start - remainder + ui + offset)
    n_symbols  = int((t[-1] - first_crossing) / ui)
    decisions  = []
    for n in range(n_symbols):
        t_center = first_crossing + (n + 0.5) * ui
        c_idx    = int(round((t_center - t[0]) / dt))
        if c_idx >= len(corrected) or c_idx < 0:
            continue
        v_sample = corrected[c_idx]
        decided  = levels_sorted[np.argmin(np.abs(levels_sorted - v_sample))]
        decisions.append(decided)
        correction = tap_weight * (decided - mid_level)
        for tap in range(1, n_taps + 1):
            t_ws  = first_crossing + (n + tap) * ui
            t_we  = first_crossing + (n + tap + 1) * ui
            s_idx = max(0, min(int(round((t_ws - t[0]) / dt)), len(corrected)))
            e_idx = max(0, min(int(round((t_we - t[0]) / dt)), len(corrected)))
            if s_idx < len(corrected):
                corrected[s_idx:e_idx] -= correction
    return corrected, np.array(decisions)


def run_fold_and_measure(t_uni, v, ui, v_refs, v_levels, offset=None, label=""):
    if offset is None:
        offset = get_phase_offset(t_uni, v, ui)
    phase      = (t_uni - offset) % ui
    H, xe, ye  = np.histogram2d(
        phase, v, bins=[300, 300],
        range=[[0, ui], [v.min() - 0.05, v.max() + 0.05]]
    )
    H = H.T
    (eye_heights, center_times, max_zeros, min_ones,
     eye_widths, center_voltages, width_ls, width_rs) = measure_eye_metrics(
        phase, v, v_refs, v_levels, ui)
    valid_eh = [eh for eh in eye_heights if eh > 0]
    valid_ew = [ew for ew in eye_widths  if ew > 0]
    min_eh   = min(valid_eh) if valid_eh else 0
    min_ew   = min(valid_ew) if valid_ew else 0
    if label:
        print(f"    [{label}] EH={min_eh*1000:.1f}mV  EW={min_ew*1e12:.1f}ps  offset={offset*1e12:.1f}ps")
    return (H, ye, phase, offset,
            eye_heights, eye_widths,
            center_times, max_zeros, min_ones,
            center_voltages, width_ls, width_rs,
            min_eh, min_ew)

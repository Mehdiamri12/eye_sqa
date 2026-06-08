"""
test_dfe.py — CTLE + DFE equalization with full visualization (PAM-Agnostic)

Pipeline:
    parse -> interpolate -> PAM Detect -> CTLE -> Phase Align -> DFE -> fold -> metrics
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eye_diagram_sqa.parser import parse_spice_txt
from eye_diagram_sqa.interpolate import interpolate_waveform
from eye_diagram_sqa.metrics import measure_eye_metrics
import settings

# ── Output folder ─────────────────────────────────────────────────────────
OUT_DIR = "dfe_results"
os.makedirs(OUT_DIR, exist_ok=True)

def save(filename):
    path = os.path.join(OUT_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#121212')
    plt.close()
    print(f"  Saved: {path}")

# ── Parameters ────────────────────────────────────────────────────────────
UI = 100
BAUD_RATE = 1/UI * 1e12  # 20 Gbaud -> 50 ps UI

# CTLE
BOOST_DB_SWEEP = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# DFE
N_TAPS     = 1
TAP_WEIGHT = 0.05

# ═══════════════════════════════════════════════════════════════════════════
# DYNAMIC PAM DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def get_pam_config(v_signal, filepath):
    """Dynamically sets v_levels and v_refs based on the filename."""
    filename = str(filepath).lower()
    if "pam4" in filename: pam_order = 4
    elif "pam3" in filename: pam_order = 3
    else: pam_order = 2  
        
    v_min = np.min(v_signal)
    v_max = np.max(v_signal)
    
    v_levels = list(np.linspace(v_min, v_max, pam_order))
    v_refs = [(v_levels[i] + v_levels[i+1]) / 2.0 for i in range(pam_order - 1)]
    
    return pam_order, v_levels, v_refs


# ═══════════════════════════════════════════════════════════════════════════
# CTLE & CLOCK RECOVERY
# ═══════════════════════════════════════════════════════════════════════════

def apply_ctle(t, v, ui, boost_db):
    dt   = np.median(np.diff(t))
    dc   = np.mean(v)
    v_ac = v - dc

    V = np.fft.rfft(v_ac)
    f = np.fft.rfftfreq(len(v), d=dt)

    f_nyq   = 1 / (2 * ui)
    boost   = 10 ** (boost_db / 20)
    f_ratio = f / (f_nyq + 1e-30)
    S       = f_ratio**2 / (1 + f_ratio**2)**2
    H       = 1 + (boost - 1) * 4 * S

    v_eq = np.fft.irfft(V * H, n=len(v)) + dc
    return v_eq, f, V, np.fft.rfft(v_eq - dc), H

def get_phase_offset(t_uni, v, ui):
    """Finds the true crossing offset using the derivative method."""
    dv = np.abs(np.diff(v))
    t_folded = t_uni[:-1] % ui
    hist, bin_edges = np.histogram(t_folded, bins=100, weights=dv)
    max_bin = np.argmax(hist)
    return (bin_edges[max_bin] + bin_edges[max_bin+1]) / 2.0


# ═══════════════════════════════════════════════════════════════════════════
# DFE (PHASE-AWARE FIX)
# ═══════════════════════════════════════════════════════════════════════════

def apply_dfe(t, v, ui, levels, offset, n_taps=1, tap_weight=0.05):
    """
    Symbol-by-symbol DFE correctly synchronized to the signal's crossing offset.
    """
    dt = np.median(np.diff(t))
    corrected = v.copy()
    levels_sorted = np.sort(levels)
    mid_level = (levels_sorted[0] + levels_sorted[-1]) / 2.0

    # Sync finding the very first valid symbol boundary
    t_start = t[0]
    remainder = t_start % ui
    if remainder <= offset:
        first_crossing = t_start - remainder + offset
    else:
        first_crossing = t_start - remainder + ui + offset

    n_symbols = int((t[-1] - first_crossing) / ui)
    decisions = []

    for n in range(n_symbols):
        # Sample exactly halfway between synced crossings
        t_center = first_crossing + (n + 0.5) * ui
        c_idx = int(round((t_center - t[0]) / dt))
        
        if c_idx >= len(corrected): break
        if c_idx < 0: continue

        v_sample = corrected[c_idx]
        decided = levels_sorted[np.argmin(np.abs(levels_sorted - v_sample))]
        decisions.append(decided)

        correction = tap_weight * (decided - mid_level)
        
        # Apply correction to the perfectly synced boundaries
        for tap in range(1, n_taps + 1):
            t_window_start = first_crossing + (n + tap) * ui
            t_window_end   = first_crossing + (n + tap + 1) * ui
            
            s_idx = int(round((t_window_start - t[0]) / dt))
            e_idx = int(round((t_window_end - t[0]) / dt))
            
            s_idx = max(0, min(s_idx, len(corrected)))
            e_idx = max(0, min(e_idx, len(corrected)))
            
            if s_idx < len(corrected):
                corrected[s_idx:e_idx] -= correction

    return corrected, np.array(decisions)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline helper — fold + measure
# ═══════════════════════════════════════════════════════════════════════════

def run_fold_and_measure(t_uni, v, ui, v_refs, v_levels, offset=None, label=""):
    
    # If no offset is provided, calculate it. Otherwise, use the forced offset.
    if offset is None:
        offset = get_phase_offset(t_uni, v, ui)

    # Fold
    phase = (t_uni - offset) % ui

    # Histogram
    resolution = 300
    H, xe, ye = np.histogram2d(
        phase, v,
        bins=[resolution, resolution],
        range=[[0, ui], [v.min()-0.05, v.max()+0.05]]
    )
    H = H.T

    # Measure
    (eye_heights, center_times, max_zeros, min_ones,
     eye_widths, center_voltages, width_ls, width_rs) = measure_eye_metrics(
        phase, v, v_refs, v_levels, ui)

    # Calculate worst-case metrics
    valid_eh = [eh for eh in eye_heights if eh > 0]
    valid_ew = [ew for ew in eye_widths if ew > 0]
    min_eh = min(valid_eh) if valid_eh else 0
    min_ew = min(valid_ew) if valid_ew else 0

    if label:
        print(f"    [{label}] Worst Eye: EH={min_eh*1000:.1f}mV  EW={min_ew*1e12:.1f}ps  offset={offset*1e12:.1f}ps")

    return (H, ye, phase, offset,
            eye_heights, eye_widths,
            center_times, max_zeros, min_ones,
            center_voltages, width_ls, width_rs,
            min_eh, min_ew)


# ═══════════════════════════════════════════════════════════════════════════
# Plot functions
# ═══════════════════════════════════════════════════════════════════════════

def plot_eye_with_metrics(H, ye, ui, phase, v, v_refs, v_levels,
                           eye_heights, eye_widths,
                           center_times, max_zeros, min_ones,
                           center_voltages, width_ls, width_rs,
                           title, filename):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='#121212')
    ax.set_facecolor('#121212')
    
    H_render = H.copy()
    H_render[H_render < 4] = 0
    
    ax.imshow(np.log1p(H_render), origin='lower', aspect='auto',
              extent=[0, ui*1e12, ye[0], ye[-1]], cmap='magma')
              
    for i, ref in enumerate(v_refs):
        ax.axhline(ref, color='red', lw=0.8, linestyle='--', alpha=0.6, label='V_Ref' if i==0 else "")
    for i, lvl in enumerate(v_levels):
        ax.axhline(lvl, color='cyan', lw=0.6, linestyle='--', alpha=0.35, label='V_Level' if i==0 else "")

    for i, (eh, ct, mz, mo) in enumerate(zip(eye_heights, center_times, max_zeros, min_ones)):
        if mo is not None and mz is not None and eh > 0:
            ax.annotate('', xy=(ct*1e12, mo), xytext=(ct*1e12, mz), arrowprops=dict(arrowstyle='<->', color='lime', lw=2))
            ax.text(ct*1e12, (mo+mz)/2, f' {eh*1000:.1f}mV', color='lime', ha='left', va='center', fontsize=10, fontweight='bold')

    for i, (ew, cv, wl, wr) in enumerate(zip(eye_widths, center_voltages, width_ls, width_rs)):
        if wl is not None and wr is not None and ew > 0:
            ax.annotate('', xy=(wr*1e12, cv), xytext=(wl*1e12, cv), arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
            ax.text((wl+wr)/2*1e12, cv, f' {ew*1e12:.1f}ps', color='cyan', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Phase within UI (ps)', color='white')
    ax.set_ylabel('Voltage (V)', color='white')
    ax.set_title(title, color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#444')
    ax.legend(fontsize=8, facecolor='#333', labelcolor='white', loc='upper right')
    save(filename)

def plot_ctle_sweep(results, optimal_db, eh_base):
    boosts  = [r[0] for r in results]
    heights = [r[1]*1000 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor='#121212')
    ax.set_facecolor('#121212')
    ax.plot(boosts, heights, 'o-', color='cyan', linewidth=2, markersize=8)
    ax.axhline(eh_base*1000, color='gray', linestyle='--', linewidth=1, label=f'Baseline {eh_base*1000:.1f}mV')
    ax.axvline(optimal_db, color='lime', linestyle='--', linewidth=1, label=f'Optimal {optimal_db}dB')
    ax.scatter([optimal_db], [heights[boosts.index(optimal_db)]], color='lime', s=120, zorder=5)
    
    ax.set_xlabel('CTLE Boost (dB)', color='white')
    ax.set_ylabel('Worst Eye Height (mV)', color='white')
    ax.set_title('CTLE Sweep — Optimal Boost (Min Eye Height)', color='white', fontsize=13)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#333', labelcolor='white')
    ax.grid(alpha=0.2)
    ax.spines[:].set_color('#444')
    save('ctle_sweep.png')

def plot_dfe_bits(t, v_before, v_after, decisions, ui, offset, v_refs, v_levels, n_bits=12):
    dt  = np.median(np.diff(t))
    mid = (np.min(v_levels) + np.max(v_levels)) / 2.0

    t_start = t[0] + 5 * ui
    t_end   = t_start + n_bits * ui
    mask    = (t >= t_start) & (t <= t_end)
    t_ps    = t[mask] * 1e12

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True, facecolor='#121212')
    fig.suptitle(f'DFE Effect on {n_bits} Bits  |  tap_weight={TAP_WEIGHT}  n_taps={N_TAPS}', color='white', fontsize=13)

    # Sync grid lines with correct offset
    first_crossing = t[0] - (t[0] % ui) + offset
    if first_crossing < t[0]: first_crossing += ui

    for ax, v, label, color in [(ax1, v_before, 'After CTLE (before DFE)', 'steelblue'), (ax2, v_after,  'After DFE', 'tomato')]:
        ax.plot(t_ps, v[mask], color=color, linewidth=1.0)
        
        for ref in v_refs: ax.axhline(ref, color='red', lw=0.8, linestyle='--', alpha=0.6)
        for lvl in v_levels: ax.axhline(lvl, color='cyan', lw=0.6, linestyle='--', alpha=0.4)
        ax.axhline(mid, color='white', lw=0.4, linestyle=':', alpha=0.3)
        
        ax.set_ylabel('Voltage (V)', color='white')
        ax.set_title(label, color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#1a1a1a')
        ax.grid(alpha=0.12)
        ax.spines[:].set_color('#444')
        ax.set_ylim(v_before.min()-0.05, v_before.max()+0.05)

        for k in range(n_bits + 6):
            ax.axvline((first_crossing + (5+k)*ui)*1e12, color='white', lw=0.3, linestyle=':', alpha=0.2)

    sym_start = 5 
    for n in range(sym_start, sym_start + n_bits):
        if n >= len(decisions): break
        t_center = first_crossing + (n + 0.5)*ui
        t_center_ps = t_center * 1e12

        for ax, v in [(ax1, v_before), (ax2, v_after)]:
            c_idx = int(round((t_center - t[0]) / dt))
            if 0 <= c_idx < len(v):
                ax.plot(t_center_ps, v[c_idx], 'o', color='yellow', markersize=4, zorder=5)

        if n > 0 and n < len(decisions):
            prev_dec   = decisions[n-1]
            correction = TAP_WEIGHT * (prev_dec - mid)
            if abs(correction) > 1e-4:
                idx = int(round((t_center - t[0]) / dt))
                if 0 <= idx < len(v_before):
                    v_orig = v_before[idx]
                    v_corr = v_after[idx]
                    if abs(v_orig - v_corr) > 1e-4:
                        ax2.annotate('', xy=(t_center_ps, v_corr), xytext=(t_center_ps, v_orig),
                                     arrowprops=dict(arrowstyle='->', color='lime', lw=1.5))

    ax2.set_xlabel('Time (ps)', color='white')
    save('dfe_bits.png')

def plot_combined(results_list, ui, v_refs, v_levels):
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(8*n, 7), facecolor='#121212')
    fig.suptitle('Equalization Pipeline — Baseline -> CTLE -> CTLE+DFE', color='white', fontsize=14)

    for ax, data in zip(axes, results_list):
        # Unpack all the metrics data
        (label, H, ye, min_eh, min_ew, 
         eh_list, ew_list, ct_list, mz_list, mo_list, cv_list, wl_list, wr_list) = data
        
        # Clean up scatter noise
        H_render = H.copy()
        H_render[H_render < 4] = 0

        ax.imshow(np.log1p(H_render), origin='lower', aspect='auto', extent=[0, ui*1e12, ye[0], ye[-1]], cmap='magma')
        
        for ref in v_refs: ax.axhline(ref, color='red', lw=0.8, linestyle='--', alpha=0.5)
        for lvl in v_levels: ax.axhline(lvl, color='cyan', lw=0.5, linestyle='--', alpha=0.3)
        
        # Draw Vertical Eye Height Arrows
        for eh, ct, mz, mo in zip(eh_list, ct_list, mz_list, mo_list):
            if mo is not None and mz is not None and eh > 0:
                ax.annotate('', xy=(ct*1e12, mo), xytext=(ct*1e12, mz), arrowprops=dict(arrowstyle='<->', color='lime', lw=2))
                ax.text(ct*1e12, (mo+mz)/2, f' {eh*1000:.1f}mV', color='lime', ha='left', va='center', fontsize=9, fontweight='bold')

        # Draw Horizontal Eye Width Arrows
        for ew, cv, wl, wr in zip(ew_list, cv_list, wl_list, wr_list):
            if wl is not None and wr is not None and ew > 0:
                ax.annotate('', xy=(wr*1e12, cv), xytext=(wl*1e12, cv), arrowprops=dict(arrowstyle='<->', color='cyan', lw=2))
                ax.text((wl+wr)/2*1e12, cv, f' {ew*1e12:.1f}ps', color='cyan', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_title(f'{label}\nWorst EH={min_eh*1000:.1f}mV  EW={min_ew*1e12:.1f}ps', color='white', fontsize=11)
        ax.set_xlabel('Phase within UI (ps)', color='white')
        ax.set_ylabel('Voltage (V)', color='white')
        ax.tick_params(colors='white')
        ax.set_facecolor('#121212')

    plt.tight_layout()
    save('combined_eye.png')


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  CTLE + DFE Equalization Test (PAM Agnostic)")
    print("=" * 60)
    print(f"  File:    {settings.PATH}")
    print("=" * 60)

    # ── 1. Parse & interpolate ─────────────────────────────────────
    print("\n[1] Parsing and interpolating ...")
    t_raw, v_raw = parse_spice_txt(settings.PATH)
    t_uni, v_int, ui = interpolate_waveform(t_raw, v_raw, BAUD_RATE)
    
    pam_order, v_levels, v_refs = get_pam_config(v_int, settings.PATH)
    
    print(f"    {len(t_uni)} points | dt={(t_uni[1]-t_uni[0])*1e12:.3f}ps | UI={ui*1e12:.0f}ps")
    print(f"    Detected PAM-{pam_order} | Levels: {[round(l,3) for l in v_levels]}")

    # ── 2. Baseline: clock recovery -> fold -> measure ───────────────
    print("\n[2] Baseline measurement ...")
    (H_base, ye_base, ph_base, off_base,
     eh_base_list, ew_base_list,
     ct_base, mz_base, mo_base,
     cv_base, wl_base, wr_base,
     eh_base, ew_base) = run_fold_and_measure(t_uni, v_int, ui, v_refs, v_levels, label="Baseline")

    plot_eye_with_metrics(
        H_base, ye_base, ui, ph_base, v_int, v_refs, v_levels,
        eh_base_list, ew_base_list, ct_base, mz_base, mo_base, cv_base, wl_base, wr_base,
        f'Baseline  Worst EH={eh_base*1000:.1f}mV  EW={ew_base*1e12:.1f}ps', 'baseline_eye.png')

    # ── 3. CTLE sweep ─────────────────────────────────────────────
    print("\n[3] CTLE sweep (clock recovery applied per boost) ...")
    sweep_results = []
    for db in BOOST_DB_SWEEP:
        v_eq, _, _, _, _ = apply_ctle(t_uni, v_int, ui, db)
        _, dyn_levels, dyn_refs = get_pam_config(v_eq, settings.PATH)
        (_, _, _, _, _, _, _, _, _, _, _, _, min_eh, min_ew) = run_fold_and_measure(t_uni, v_eq, ui, dyn_refs, dyn_levels)
        sweep_results.append((db, min_eh, min_ew))
        print(f"    boost={db:>3}dB -> Worst EH={min_eh*1000:.1f}mV  EW={min_ew*1e12:.1f}ps")

    best_idx    = int(np.argmax([r[1] for r in sweep_results]))
    optimal_db  = sweep_results[best_idx][0]
    print(f"  Optimal boost: {optimal_db}dB -> Worst EH={sweep_results[best_idx][1]*1000:.1f}mV")

    plot_ctle_sweep(sweep_results, optimal_db, eh_base)

    # ── 4. Apply optimal CTLE ─────────────────────────────────────
    print(f"\n[4] Applying Optimal CTLE (boost={optimal_db}dB) ...")
    v_ctle, _, _, _, _ = apply_ctle(t_uni, v_int, ui, optimal_db)
    _, ctle_levels, ctle_refs = get_pam_config(v_ctle, settings.PATH)

    # Get the official offset from the clean CTLE wave to use everywhere else
    ctle_offset = get_phase_offset(t_uni, v_ctle, ui)

    (H_ctle, ye_ctle, ph_ctle, off_ctle,
     eh_ctle_list, ew_ctle_list, ct_ctle, mz_ctle, mo_ctle, cv_ctle, wl_ctle, wr_ctle, eh_ctle, ew_ctle) = run_fold_and_measure(
        t_uni, v_ctle, ui, ctle_refs, ctle_levels, offset=ctle_offset, label="After CTLE")

    plot_eye_with_metrics(
        H_ctle, ye_ctle, ui, ph_ctle, v_ctle, ctle_refs, ctle_levels,
        eh_ctle_list, ew_ctle_list, ct_ctle, mz_ctle, mo_ctle, cv_ctle, wl_ctle, wr_ctle,
        f'After CTLE ({optimal_db}dB)  Worst EH={eh_ctle*1000:.1f}mV  EW={ew_ctle*1e12:.1f}ps', 'ctle_eye.png')

    # ── 5. Apply DFE ─────────────────────────────────────────────
    print(f"\n[5] Applying DFE (n_taps={N_TAPS}, tap_weight={TAP_WEIGHT}) ...")
    # Feed the CTLE offset INTO the DFE so it samples at the actual eye center
    v_dfe, decisions = apply_dfe(t_uni, v_ctle, ui, ctle_levels, offset=ctle_offset, n_taps=N_TAPS, tap_weight=TAP_WEIGHT)
    _, dfe_levels, dfe_refs = get_pam_config(v_dfe, settings.PATH)

    # Fold the DFE using the EXACT same offset so the visual doesn't shift
    (H_dfe, ye_dfe, ph_dfe, off_dfe,
     eh_dfe_list, ew_dfe_list, ct_dfe, mz_dfe, mo_dfe, cv_dfe, wl_dfe, wr_dfe, eh_dfe, ew_dfe) = run_fold_and_measure(
        t_uni, v_dfe, ui, dfe_refs, dfe_levels, offset=ctle_offset, label="After CTLE+DFE")

    # ── 6. DFE per-bit plot ───────────────────────────────────────
    print("\n[6] DFE per-bit visualization ...")
    plot_dfe_bits(t_uni, v_ctle, v_dfe, decisions, ui, ctle_offset, dfe_refs, dfe_levels, n_bits=12)

    # ── 7. DFE eye diagram ────────────────────────────────────────
    print("\n[7] DFE eye diagram ...")
    plot_eye_with_metrics(
        H_dfe, ye_dfe, ui, ph_dfe, v_dfe, dfe_refs, dfe_levels,
        eh_dfe_list, ew_dfe_list, ct_dfe, mz_dfe, mo_dfe, cv_dfe, wl_dfe, wr_dfe,
        f'After CTLE+DFE  Worst EH={eh_dfe*1000:.1f}mV  EW={ew_dfe*1e12:.1f}ps', 'dfe_eye.png')

    # ── 8. Combined comparison ────────────────────────────────────
    print("\n[8] Combined comparison ...")
    plot_combined([
        ('Baseline', H_base, ye_base, eh_base, ew_base, 
         eh_base_list, ew_base_list, ct_base, mz_base, mo_base, cv_base, wl_base, wr_base),
         
        (f'CTLE {optimal_db}dB', H_ctle, ye_ctle, eh_ctle, ew_ctle, 
         eh_ctle_list, ew_ctle_list, ct_ctle, mz_ctle, mo_ctle, cv_ctle, wl_ctle, wr_ctle),
         
        ('CTLE+DFE', H_dfe,  ye_dfe,  eh_dfe,  ew_dfe, 
         eh_dfe_list, ew_dfe_list, ct_dfe, mz_dfe, mo_dfe, cv_dfe, wl_dfe, wr_dfe),
    ], ui, dfe_refs, dfe_levels)

    # ── 9. Final report ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FINAL REPORT (Worst-Case Eyes)")
    print(f"{'='*60}")
    print(f"  {'Stage':<22} {'EH(mV)':>8} {'EW(ps)':>8} {'ΔEH':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline':<22} {eh_base*1000:>8.1f} {ew_base*1e12:>8.1f} {'—':>8}")
    print(f"  {f'CTLE ({optimal_db}dB)':<22} {eh_ctle*1000:>8.1f} {ew_ctle*1e12:>8.1f} {(eh_ctle-eh_base)*1000:>+8.1f}")
    print(f"  {'CTLE+DFE':<22} {eh_dfe*1000:>8.1f} {ew_dfe*1e12:>8.1f} {(eh_dfe-eh_ctle)*1000:>+8.1f}")
    print(f"  {'Total':<22} {eh_dfe*1000:>8.1f} {ew_dfe*1e12:>8.1f} {(eh_dfe-eh_base)*1000:>+8.1f}")
    print(f"{'='*60}")
    print(f"\n  Outputs: {OUT_DIR}/")
    print("Done.")

if __name__ == "__main__":
    main()
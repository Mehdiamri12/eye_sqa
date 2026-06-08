import numpy as np
from scipy.interpolate import interp1d

def parse_spice_txt(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    times, voltages = [], []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and not line[0].replace('-','').replace('+','')[0].isdigit():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                times.append(float(parts[0]))
                voltages.append(float(parts[1]))
            except ValueError:
                continue
    t = np.array(times)
    v = np.array(voltages)
    assert len(t) >= 2
    assert np.all(np.diff(t) >= 0)
    return t, v
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import settings

    t, v = parse_spice_txt(settings.PATH)

    # --- printed checks ---
    print(f"Points:        {len(t)}")
    print(f"Time start:    {t[0]*1e12:.3f} ps")
    print(f"Time end:      {t[-1]*1e9:.3f} ns")
    print(f"Min timestep:  {np.diff(t).min()*1e15:.3f} fs")
    print(f"Max timestep:  {np.diff(t).max()*1e12:.3f} ps")
    print(f"Voltage min:   {v.min():.4f} V")
    print(f"Voltage max:   {v.max():.4f} V")
    print(f"Voltage mean:  {v.mean():.4f} V")
    print()
    print("First 5 points:")
    for i in range(5):
        print(f"  t={t[i]*1e12:12.4f} ps   V={v[i]:.6f} V")
    print("Last 5 points:")
    for i in range(-5, 0):
        print(f"  t={t[i]*1e9:12.6f} ns   V={v[i]:.6f} V")

    # --- visual check ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    fig.suptitle("Parser output check — netlist_pam2.txt")

    # full waveform
    ax1.plot(t * 1e9, v, linewidth=0.5, color='steelblue')
    ax1.set_xlabel("Time (ns)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title("Full waveform (all 11,610 points)")
    ax1.grid(True, alpha=0.3)

    # zoom into first 5 UIs (first 250ps)
    mask = t <= 250e-12
    ax2.plot(t[mask] * 1e12, v[mask], linewidth=1.0,
             color='tomato', marker='o', markersize=2)
    ax2.set_xlabel("Time (ps)")
    ax2.set_ylabel("Voltage (V)")
    ax2.set_title("First 250ps — notice non-uniform SPICE timesteps")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("eye diagram/check_parser.png", dpi=150)
    plt.show()
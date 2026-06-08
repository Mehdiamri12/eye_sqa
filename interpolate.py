import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from parser import parse_spice_txt
import settings

def interpolate_waveform(t, v, baud_rate, points_per_ui=200):
    ui = 1.0 / baud_rate
    dt = ui / points_per_ui
    t_uniform = np.arange(t[0], t[-1], dt)
    
    cs = CubicSpline(t, v)
    v_interp = cs(t_uniform)
    
    return t_uniform, v_interp, ui

if __name__ == "__main__":
    # 1. Parse
    t_raw, v_raw = parse_spice_txt(settings.PATH)
    
    # 2. Interpolate
    t_uni, v_int, ui = interpolate_waveform(t_raw, v_raw, baud_rate=20e9)
    
    # 3. Simple Visual Check
    plt.figure(figsize=(10, 4))
    # Plotting first 500ps to see the smooth interpolation
    mask = t_uni <= 500e-12
    plt.plot(t_uni[mask]*1e12, v_int[mask], label='Cubic Spline', color='blue')
    plt.scatter(t_raw[t_raw <= 500e-12]*1e12, v_raw[t_raw <= 500e-12], 
                color='red', s=10, label='Raw Points')
    
    plt.title("Interpolation Check (Cubic Spline)")
    plt.xlabel("Time (ps)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    
    # 4. Save to the existing folder
    plt.savefig("eye diagram/check_interpolation.png")
    print("Interpolation plot saved to 'eye diagram/check_interpolation.png'")
    
    # Optional: plt.show()
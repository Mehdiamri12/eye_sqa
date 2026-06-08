import numpy as np

def get_overshoot(v_int, v_high=1.2, v_low=0.6):
    v_max = v_int.max()
    v_min = v_int.min()
    overshoot  = v_max - v_high
    undershoot = v_low - v_min
    return overshoot, undershoot

def get_eye_height(phase, v_int, v_refs, ui, n_slices=70):
    all_best_heights  = []
    all_best_t        = []
    all_best_min_one  = []
    all_best_max_zero = []

    for v_ref in v_refs:
        best_height   = 0
        best_t        = 0
        best_min_one  = None
        best_max_zero = None

        slices = np.linspace(0.25 * ui, 0.75 * ui, n_slices, endpoint=False)

        for t in slices:
            mask    = (phase >= t) & (phase < t + ui / n_slices)
            v_slice = v_int[mask]
            if len(v_slice) == 0:
                continue

            above = v_slice[v_slice >= v_ref]
            below = v_slice[v_slice <  v_ref]
            if len(above) == 0 or len(below) == 0:
                continue

            min_one  = above.min()
            max_zero = below.max()
            height   = min_one - max_zero

            if height > best_height:
                best_height   = height
                best_t        = t
                best_min_one  = min_one
                best_max_zero = max_zero

        all_best_heights.append(best_height)
        all_best_t.append(best_t)
        all_best_min_one.append(best_min_one)
        all_best_max_zero.append(best_max_zero)

    return all_best_heights, all_best_t, all_best_max_zero, all_best_min_one

def get_eye_width(phase, v_int, v_levels, ui, n_slices=50, min_samples=10):
    all_best_widths    = []
    all_best_v         = []
    all_best_max_left  = []
    all_best_min_right = []

    slice_height = (v_int.max() - v_int.min()) / n_slices

    for i in range(len(v_levels) - 1):
        v_min = v_levels[i]
        v_max = v_levels[i + 1]

        best_width     = 0
        best_v         = 0
        best_max_left  = None
        best_min_right = None

        for v in np.arange(v_min, v_max, slice_height):
            mask    = (v_int >= v) & (v_int < v + slice_height)
            t_slice = phase[mask]
            if len(t_slice) < min_samples:
                continue

            left  = t_slice[t_slice <  0.5 * ui]
            right = t_slice[t_slice >= 0.5 * ui]
            if len(left) == 0 or len(right) == 0:
                continue

            width = right.min() - left.max()
            if width > best_width:
                best_width     = width
                best_v         = v + slice_height / 2
                best_max_left  = left.max()
                best_min_right = right.min()

        all_best_widths.append(best_width)
        all_best_v.append(best_v)
        all_best_max_left.append(best_max_left)
        all_best_min_right.append(best_min_right)

    return all_best_widths, all_best_v, all_best_max_left, all_best_min_right

def measure_eye_metrics(phase, v_int, v_refs, v_levels, ui):
    eye_height, center_time, max_zero, min_one = get_eye_height(
        phase, v_int, v_refs, ui)
    eye_width, center_voltage, width_l, width_r = get_eye_width(
        phase, v_int, v_levels, ui)
    return (eye_height, center_time, max_zero, min_one,
            eye_width, center_voltage, width_l, width_r)

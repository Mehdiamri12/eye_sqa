"""
api/main.py — FastAPI wrapper around the eye diagram pipeline.

Endpoints:
    GET  /health             — CI liveness check
    POST /analyze/json       — submit samples as JSON array
    POST /analyze/file       — upload a .txt SPICE file
    GET  /results/{id}       — retrieve a stored result
    GET  /                   — serve the frontend HTML

Run the server:
    uvicorn api.main:app --reload --port 8000

Both /analyze endpoints run the full pipeline:
    parse → interpolate → CTLE sweep → optimal CTLE → DFE → metrics

Response schema (identical for both endpoints):
    {
        id, modulation, pam_order, n_samples, ui_ps,
        baseline:  { eye_height_mv, eye_width_ps },
        ctle:      { optimal_boost_db, eye_height_mv, improvement_mv, sweep },
        dfe:       { eye_height_mv, n_decisions },
        status, message
    }
"""

import os
import sys
import uuid
import tempfile
import numpy as np
from typing import List

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, field_validator
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import (
    get_pam_config, apply_ctle, apply_dfe,
    get_phase_offset, run_fold_and_measure,
)

app = FastAPI(title="Eye Diagram Analysis API", version="1.0.0")


BOOST_SWEEP    = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
STORE: dict    = {}
# 1. Get the absolute path to the root 'eye_sqa' directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. Point to the frontend folder
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# 3. Tell FastAPI to serve the index.html file whenever someone goes to "/"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


# ── Request model ─────────────────────────────────────────────────────────────

class JSONRequest(BaseModel):
    samples:    List[float]
    ui_ps:      float
    modulation: str = "NRZ"

    @field_validator("samples")
    @classmethod
    def validate_samples(cls, v):
        if len(v) < 100:
            raise ValueError(f"Too few samples ({len(v)}). Minimum 100 required.")
        if any(not np.isfinite(x) for x in v):
            raise ValueError("Samples contain NaN or Inf. Check simulation for convergence failures.")
        if len(set(round(x, 6) for x in v)) == 1:
            raise ValueError("All samples identical — constant signal. No transitions to measure.")
        return v

    @field_validator("ui_ps")
    @classmethod
    def validate_ui(cls, v):
        if v <= 0:
            raise ValueError(f"ui_ps must be positive, got {v}")
        return v

    @field_validator("modulation")
    @classmethod
    def validate_mod(cls, v):
        if v not in ["NRZ", "PAM3", "PAM4"]:
            raise ValueError(f"Unsupported modulation '{v}'. Use NRZ, PAM3, or PAM4.")
        return v


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(t, v, ui, filepath_hint):
    """
    Full pipeline: baseline → CTLE sweep → optimal CTLE → DFE.
    Prints progress at every stage for easy debugging.
    Returns a result dict ready to serialize.
    """
    print(f"\n[pipeline] Starting — {len(v):,} samples | UI={ui*1e12:.0f}ps")

    _, v_levels, v_refs = get_pam_config(v, filepath_hint)
    pam_order = len(v_levels)
    print(f"[pipeline] PAM-{pam_order} | levels={[round(l,3) for l in v_levels]}")

    if not np.all(np.isfinite(v)):
        raise ValueError("Signal contains NaN/Inf — likely a SPICE convergence failure.")

    # Baseline
    print("[pipeline] Computing baseline ...")
    res_base = run_fold_and_measure(t, v, ui, v_refs, v_levels, label="Baseline")
    eh_base, ew_base = res_base[-2], res_base[-1]
    if eh_base == 0:
        raise ValueError("Baseline eye is completely closed. Check UI parameter or signal.")
    print(f"[pipeline] Baseline EH={eh_base*1000:.1f}mV  EW={ew_base*1e12:.1f}ps")

    # CTLE sweep
    print("[pipeline] Running CTLE sweep ...")
    sweep = {}
    for db in BOOST_SWEEP:
        v_eq, _, _, _, _ = apply_ctle(t, v, ui, db)
        _, lv_eq, rv_eq  = get_pam_config(v_eq, filepath_hint)
        eh_eq            = run_fold_and_measure(t, v_eq, ui, rv_eq, lv_eq)[-2]
        sweep[db]        = eh_eq
        print(f"  {db:>4}dB → EH={eh_eq*1000:.1f}mV")

    optimal_db = max(sweep, key=sweep.get)
    print(f"[pipeline] Optimal boost: {optimal_db}dB → EH={sweep[optimal_db]*1000:.1f}mV")

    # Apply optimal CTLE
    v_opt, _, _, _, _ = apply_ctle(t, v, ui, optimal_db)
    _, lv_opt, rv_opt = get_pam_config(v_opt, filepath_hint)
    ctle_offset       = get_phase_offset(t, v_opt, ui)
    res_ctle = run_fold_and_measure(t, v_opt, ui, rv_opt, lv_opt,
                                    offset=ctle_offset, label="After CTLE")
    eh_ctle  = res_ctle[-2]

    # DFE
    print("[pipeline] Applying DFE ...")
    v_dfe, decisions = apply_dfe(t, v_opt, ui, lv_opt, ctle_offset,
                                  n_taps=1, tap_weight=0.05)
    _, lv_dfe, rv_dfe = get_pam_config(v_dfe, filepath_hint)
    res_dfe = run_fold_and_measure(t, v_dfe, ui, rv_dfe, lv_dfe,
                                   offset=ctle_offset, label="After DFE")
    eh_dfe  = res_dfe[-2]
    ew_dfe  = res_dfe[-1]
    print(f"[pipeline] DFE: EH={eh_dfe*1000:.1f}mV  decisions={len(decisions)}")

    print(f"[pipeline] Done — Baseline→CTLE→DFE: "
          f"{eh_base*1000:.0f}→{eh_ctle*1000:.0f}→{eh_dfe*1000:.0f}mV")

    return {
        "pam_order": pam_order,
        "n_samples": len(v),
        "ui_ps":     round(ui * 1e12, 2),
        "baseline":  {"eye_height_mv": round(eh_base*1000, 2),
                      "eye_width_ps":  round(ew_base*1e12, 2)},
        "ctle":      {"optimal_boost_db": optimal_db,
                      "eye_height_mv":    round(eh_ctle*1000, 2),
                      "improvement_mv":   round((eh_ctle-eh_base)*1000, 2),
                      "sweep":            {str(k): round(e*1000, 2)
                                           for k, e in sweep.items()}},
        "dfe":       {"eye_height_mv": round(eh_dfe*1000, 2),
                      "eye_width_ps":  round(ew_dfe*1e12, 2),
                      "n_decisions":   len(decisions)},
        "status":    "success",
        "message":   (f"PAM-{pam_order} | optimal CTLE {optimal_db}dB | "
                      f"EH {eh_base*1000:.0f}→{eh_ctle*1000:.0f}→{eh_dfe*1000:.0f}mV"),
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    print("[api] GET /health")
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze/json", status_code=201)
def analyze_json(req: JSONRequest):
    print(f"\n[api] POST /analyze/json — {len(req.samples)} samples | "
          f"ui={req.ui_ps}ps | mod={req.modulation}")
    try:
        v  = np.array(req.samples)
        ui = req.ui_ps * 1e-12
        t  = np.arange(len(v)) * (ui / 20)
        result = run_pipeline(t, v, ui, f"{req.modulation.lower()}_signal.txt")
        result["id"]         = str(uuid.uuid4())[:8]
        result["modulation"] = req.modulation
        STORE[result["id"]]  = result
        print(f"[api] Result stored — id={result['id']}")
        return result
    except ValueError as e:
        print(f"[api] 422 — {e}")
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        print(f"[api] 500 — {e}")
        raise HTTPException(500, detail=f"Pipeline error: {e}")


@app.post("/analyze/file", status_code=201)
async def analyze_file(file: UploadFile = File(...)):
    print(f"\n[api] POST /analyze/file — {file.filename}")

    if not file.filename.endswith(".txt"):
        raise HTTPException(422, detail=f"Expected .txt SPICE file, got: {file.filename}")

    try:
        from parser import parse_spice_txt
        from interpolate import interpolate_waveform
        import re

        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="wb") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            t_raw, v_raw = parse_spice_txt(tmp_path)
            print(f"[api] Parsed {len(t_raw):,} samples from {file.filename}")
        finally:
            os.unlink(tmp_path)

        m      = re.search(r"(\d+)_ui", file.filename.lower())
        ui_ps  = float(m.group(1)) if m else 80.0
        baud   = 1 / ui_ps * 1e12
        t, v, ui = interpolate_waveform(t_raw, v_raw, baud)
        print(f"[api] Interpolated — UI={ui_ps:.0f}ps | {len(t):,} pts")

        result               = run_pipeline(t, v, ui, file.filename)
        result["id"]         = str(uuid.uuid4())[:8]
        result["modulation"] = ("PAM4" if "pam4" in file.filename.lower()
                                else "PAM3" if "pam3" in file.filename.lower()
                                else "NRZ")
        result["filename"]   = file.filename
        STORE[result["id"]]  = result
        print(f"[api] Result stored — id={result['id']}")
        return result

    except HTTPException:
        raise
    except ValueError as e:
        print(f"[api] 422 — {e}")
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        print(f"[api] 500 — {e}")
        raise HTTPException(500, detail=f"Pipeline error: {e}")


@app.get("/results/{result_id}")
def get_result(result_id: str):
    print(f"[api] GET /results/{result_id}")
    if result_id not in STORE:
        raise HTTPException(404, detail=f"Result '{result_id}' not found.")
    return STORE[result_id]


@app.get("/")
def frontend():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "frontend", "index.html")
    return FileResponse(path) if os.path.exists(path) else {"message": "API running — see /docs"}

# Eye Diagram SQA

Full SQA test suite for the CTLE+DFE eye diagram pipeline.
Tests SPICE data parsing, CTLE sweep correctness, and the web UI.

## Setup

```bash
pip install -r requirements.txt
playwright install chromium
```

Copy into this folder:
- `parser.py`, `interpolate.py`, `metrics.py`, `settings.py`, `test.py`
- `Data/` folder with your SPICE files

## Run tests

```bash
# All unit + API tests (no server needed)
pytest tests/test_pipeline.py tests/test_api.py -v -s

# Unit tests only
pytest tests/test_pipeline.py -v -s

# API tests only
pytest tests/test_api.py -v -s

# E2E tests (browser opens automatically)
pytest tests/test_e2e.py -v -s

# Single test
pytest tests/test_pipeline.py::TestCTLESweep::test_pam3_cliff_is_detected -v -s
```

## Run the UI manually

```bash
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000
```

## Test count

| File              | Tests | What it protects                              |
|-------------------|-------|-----------------------------------------------|
| test_pipeline.py  | 9     | Parser robustness + CTLE sweep on real SPICE  |
| test_api.py       | 6     | Golden master metrics + input guards          |
| test_e2e.py       | 3     | Danger zone rendering + error display         |
| **Total**         | **18**| **Full pipeline from file to browser**        |

## How to break the tests (for demo)

### Wrong dB formula (test_sweep_finds_correct_optimal_boost)
In `pipeline.py` → `apply_ctle`:
```python
# Change:
boost = 10 ** (boost_db / 20)
# To:
boost = 10 ** (boost_db / 10)
```
Effect: sweep curve shifts left, wrong optimal reported for both channels.

### Broken PAM3 detection (test_pam3_cliff_is_detected)
In `pipeline.py` → `get_pam_config`:
```python
# Change:
elif "pam3" in filename: pam_order = 3
# To:
elif "pam3x" in filename: pam_order = 3
```
Effect: PAM3 file processed as PAM2, all sweep values collapse to ~0mV.

### Inverted CTLE filter (test_optimal_boost_improves_baseline)
In `pipeline.py` → `apply_ctle`:
```python
# Change:
H = 1 + (boost - 1) * 4 * S
# To:
H = 1 - (boost - 1) * 4 * S
```
Effect: CTLE attenuates instead of boosts, EH decreases after equalization.

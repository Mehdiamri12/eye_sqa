from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# Data folder relative to this file
BASE_DIR = _THIS_DIR / "Data"


# Auto-find all netlist_pam*.txt files across all subdirectories
def find_all_files():
    files = sorted(BASE_DIR.rglob("*netlist_pam*.txt"))

    # Fallback: also check next to this file if Data is empty/missing
    if not files:
        files = sorted(_THIS_DIR.glob("netlist_pam*.txt"))

    return files


ALL_FILES = find_all_files()

PATH = ALL_FILES[0] if ALL_FILES else None


if __name__ == "__main__":
    print(f"settings.py directory: {_THIS_DIR}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Found {len(ALL_FILES)} file(s):")
    for f in ALL_FILES:
        print(f"  {f}")
    print(f"PATH: {PATH}")
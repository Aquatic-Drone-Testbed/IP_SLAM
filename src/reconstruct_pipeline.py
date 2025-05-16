# reconstruct_pipeline.py

import time
import shutil
import re
from pathlib import Path
from reconstruct_scan_single import reconstruct_scan_single

# ── CONFIG ─────────────────────────────────────────────────────────────────────
RAW_DATA_DIR    = Path("scans/raw_dumps")        # incoming .txt scans
PROCESSED_DIR   = Path("scans/processed_dumps")  # move here after processing
VALID_ROOT      = Path("scans/valid_frames")     # root for outputs
POLL_INTERVAL   = 10  # seconds between checks
file_id_start = 1
# ────────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in (RAW_DATA_DIR, PROCESSED_DIR, VALID_ROOT):
        d.mkdir(parents=True, exist_ok=True)
    for sub in ("polar", "cartesian", "heatmap"):
        (VALID_ROOT / sub).mkdir(parents=True, exist_ok=True)

def extract_scan_id(stem: str) -> int:
    """Pull the first run of digits from a filename stem, or return a large dummy."""
    m = re.search(r"(\d+)", stem)
    return int(m.group(1)) if m else float('inf')

def main():
    ensure_dirs()
    seen = set()
    file_id_start = 1
    while True:
        # Gather all .txt files not yet processed
        all_txt = [p for p in RAW_DATA_DIR.glob("*.txt") if p.name not in seen]
        # Sort them by numeric scan_id ascending
        all_txt.sort(key=lambda p: extract_scan_id(p.stem))

        for txt in all_txt:
            scan_id = extract_scan_id(txt.stem)
            print(f"[reconstruct] processing {txt.name} → scan_id={scan_id:05d}")
            try:
                polar_path, cart_path, heat_path = reconstruct_scan_single(
                    file_path=str(txt),
                    output_folder=str(VALID_ROOT / "polar"),
                    polar_output_folder=str("scans/0_raw_cartesian_scans"),
                    heatmap_output_folder=str(VALID_ROOT / "heatmap"),
                    scan_id=file_id_start
                )
                if polar_path != None:
                    file_id_start +=1
                print(f"  → polar:   {polar_path}")
                print(f"  → cart:    {cart_path}")
                print(f"  → heatmap: {heat_path}")

                # mark as done and move the text file
                seen.add(txt.name)
                shutil.move(str(txt), PROCESSED_DIR / txt.name)

            except Exception as e:
                print(f"  [ERROR] reconstruct_scan_single failed on {txt.name}: {e}")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

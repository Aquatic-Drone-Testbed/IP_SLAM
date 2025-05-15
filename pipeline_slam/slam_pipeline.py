# slam_pipeline.py

import time
import traceback
import sys
import shutil
from pathlib import Path

from add_center_marker import add_center_marker
from preprocess_scans import preprocess_scans
from estimate_transformations import estimate_transformations
from orient_scans import orient_scans
from stitch_map import stitch_map
from stitch_map_pix_thresh import stitch_map_pix_thresh
from print_path import print_path
from cartesian_to_polar import cartesian_to_polar
from test_stitch2 import test_stitch2
from combine_pose_file import combine_pose_file

# ── CONFIG ─────────────────────────────────────────────────────────────────────
VALID_ROOT    = Path("data/valid_frames")
CART_ROOT     = VALID_ROOT / "cartesian"
WORK_ROOT     = Path("data/slam_work")
POLL_INTERVAL = 15  # seconds
STATE_FILE    = WORK_ROOT / "processed_frames.txt"
# ────────────────────────────────────────────────────────────────────────────────

def ensure_dirs():
    # Main workspace
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    # Per‐step folders (inputs + outputs)
    for d in (
        "01_centered",
        "02_processed",
        "04_aligned",
    ):
        (WORK_ROOT / d).mkdir(parents=True, exist_ok=True)

def load_processed():
    if not STATE_FILE.exists():
        return set()
    return set(STATE_FILE.read_text().splitlines())

def save_processed(frames):
    STATE_FILE.write_text("\n".join(sorted(frames)))

def copy_files(src_paths, dst_folder):
    for p in src_paths:
        shutil.copy(p, dst_folder / p.name)

def incremental_preprocess(new_cartesian):
    """
    Run steps 1 & 2 only on the new cartesian frames,
    appending results into WORK_ROOT/01_centered and /02_processed.
    """
    tmp_cart = WORK_ROOT / "tmp_cart"
    tmp_cent = WORK_ROOT / "tmp_center"
    tmp_proc = WORK_ROOT / "tmp_proc"
    for d in (tmp_cart, tmp_cent, tmp_proc):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()

    # Copy only the new ones into tmp_cart
    copy_files(new_cartesian, tmp_cart)

    # Step 1: center marker (tmp_cart → tmp_cent), then merge into 01_centered
    add_center_marker(str(tmp_cart), str(tmp_cent))
    copy_files(list(tmp_cent.glob("*.jpg")), WORK_ROOT / "01_centered")

    # Step 2: preprocess (tmp_cent → tmp_proc), then merge into 02_processed
    preprocess_scans(str(tmp_cent), str(tmp_proc))
    copy_files(list(tmp_proc.glob("*.jpg")), WORK_ROOT / "02_processed")

    # clean up
    shutil.rmtree(tmp_cart)
    shutil.rmtree(tmp_cent)
    shutil.rmtree(tmp_proc)

def run_full_slam():
    """
    Run steps 3–9 on the *entire* set of processed & centered frames.
    (These folders now contain both old + newly appended images.)
    """
    step1 = WORK_ROOT / "01_centered"
    step2 = WORK_ROOT / "02_processed"
    aligned = WORK_ROOT / "04_aligned"

    # Step 3: estimate transformations
    tf_file     = WORK_ROOT / "03_transformations.npy"
    match_dbg   = WORK_ROOT / "03_matches"
    overlay_dbg = WORK_ROOT / "03_overlays"
    angle_file  = WORK_ROOT / "03_angles.txt"
    estimate_transformations(
        str(step2),
        str(tf_file),
        str(match_dbg),
        str(overlay_dbg),
        str(angle_file),
        "AKAZE",
        100
    )

    # Step 4: orient scans
    orient_scans(
        str(step2),
        str(step1),
        str(aligned),
        str(tf_file),
        str(WORK_ROOT / "04_proc.gif"),
        str(WORK_ROOT / "04_orig.gif"),
        str(WORK_ROOT / "04_unoriented.gif")
    )

    # Step 5: stitch map
    stitch_map(
        str(aligned) + "_orig",
        str(WORK_ROOT / "05_map.png"),
        str(WORK_ROOT / "05_steps"),
        str(WORK_ROOT / "05_map.gif")
    )

    # Step 6: filtered stitch
    stitch_map_pix_thresh(
        str(aligned) + "_orig",
        str(WORK_ROOT / "06_map_pt.png"),
        str(WORK_ROOT / "06_map_pt_bw.png"),
        str(WORK_ROOT / "06_steps_pt"),
        str(WORK_ROOT / "06_map_pt.gif"),
        str(WORK_ROOT / "06_centers.txt"),
        pixel_threshold=150
    )

    # Step 7: print boat path
    print_path(
        str(WORK_ROOT / "05_map.png"),
        str(WORK_ROOT / "06_centers.txt"),
        str(angle_file),
        str(WORK_ROOT / "07_map_path.png"),
        str(WORK_ROOT / "07_map_path.gif")
    )

    # Step 8: revert + lidar + test_stitch2
    cartesian_to_polar(
        str(aligned) + "_orig",
        str(WORK_ROOT / "08_reverted_polar"),
        str(WORK_ROOT / "08_reverted_polar_lidarized"),
        str(WORK_ROOT / "08_lidar_cartesian")
    )
    test_stitch2(
        str(aligned) + "_orig",
        str(WORK_ROOT / "08_lidar_cartesian") + "_mask",
        str(WORK_ROOT / "08_stitched_map_test.png"),
        str(WORK_ROOT / "08_stitched_map_test_bw.png"),
        str(WORK_ROOT / "08_stitched_maps_steps_test"),
        str(WORK_ROOT / "08_stitching_process_test.gif"),
        str(WORK_ROOT / "08_centers.txt"),
        pixel_threshold=150
    )

    # Step 9: combine pose file
    combine_pose_file(
        str(WORK_ROOT / "08_centers.txt"),
        str(angle_file),
        str(WORK_ROOT / "09_pose_estimation.txt")
    )


def main():
    ensure_dirs()
    processed = load_processed()

    while True:
        # find all cartesian frames
        all_cart = sorted(CART_ROOT.glob("*.jpg"))
        all_names = {p.name for p in all_cart}
        new_names = all_names - processed
        if new_names:
            # map names back to Paths
            new_paths = [p for p in all_cart if p.name in new_names]
            print(f"[slam] found {len(new_paths)} new frames, preprocessing…")
            try:
                # 1–2: incrementally center & preprocess only new frames
                incremental_preprocess(new_paths)
                # mark them done
                processed.update(new_names)
                save_processed(processed)

                # 3–9: run SLAM on full set (old + new)
                print("[slam] running full SLAM pipeline on all frames…")
                run_full_slam()
            except Exception as e:
                print(f"  [ERROR] incremental SLAM failed: {e}", file=sys.stderr)
                traceback.print_exc()
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

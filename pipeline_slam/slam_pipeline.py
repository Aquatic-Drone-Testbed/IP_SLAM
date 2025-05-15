# slam_pipeline.py

import time
import traceback
import sys
from pathlib import Path
from datetime import datetime

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
POLL_INTERVAL = 15  # seconds between polling for new images
# ────────────────────────────────────────────────────────────────────────────────

def ensure_root():
    """Make sure the top‐level WORK_ROOT exists."""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)

def run_slam():
    # 1) create a unique timestamped folder for this run
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_folder = WORK_ROOT / f"run_{ts}"
    for step in range(1, 10):
        (run_folder / f"{step:02d}").mkdir(parents=True, exist_ok=True)

    # Step 1: add center marker
    step1 = run_folder / "01_centered"
    add_center_marker(str(CART_ROOT), str(step1))

    # Step 2: preprocess scans
    step2 = run_folder / "02_processed"
    preprocess_scans(str(step1), str(step2))

    # Step 3: estimate transformations
    tf_file     = run_folder / "03_transformations.npy"
    match_dbg   = run_folder / "03_matches"
    overlay_dbg = run_folder / "03_overlays"
    angle_file  = run_folder / "03_angles.txt"
    estimate_transformations(
        str(step2),
        str(tf_file),
        str(match_dbg),
        str(overlay_dbg),
        str(angle_file),
        "AKAZE",    # feature detector
        100         # num_good_matches
    )

    # Step 4: orient scans
    aligned = run_folder / "04_aligned"
    orient_scans(
        str(step2),
        str(step1),                       # use centered scans as "original"
        str(aligned),
        str(tf_file),
        str(run_folder / "04_proc.gif"),
        str(run_folder / "04_orig.gif"),
        str(run_folder / "04_unoriented.gif")
    )

    # Step 5: stitch map
    stitch_map(
        str(aligned) + "_orig",
        str(run_folder / "05_map.png"),
        str(run_folder / "05_steps"),
        str(run_folder / "05_map.gif")
    )

    # Step 6: stitch filtered map
    stitch_map_pix_thresh(
        str(aligned) + "_orig",
        str(run_folder / "06_map_pt.png"),
        str(run_folder / "06_map_pt_bw.png"),
        str(run_folder / "06_steps_pt"),
        str(run_folder / "06_map_pt.gif"),
        str(run_folder / "06_centers.txt"),
        pixel_threshold=150
    )

    # Step 7: print boat path
    print_path(
        str(run_folder / "05_map.png"),
        str(run_folder / "06_centers.txt"),
        str(angle_file),
        str(run_folder / "07_map_path.png"),
        str(run_folder / "07_map_path.gif")
    )

    # Step 8: revert to polar + lidar + test_stitch2
    reverted_polar = run_folder / "08_reverted_polar"
    lidar_polar    = run_folder / "08_reverted_polar_lidarized"
    lidar_cart     = run_folder / "08_lidar_cartesian"

    cartesian_to_polar(
        str(aligned) + "_orig",
        str(reverted_polar),
        str(lidar_polar),
        str(lidar_cart)
    )

    out_png   = run_folder / "08_stitched_map_test.png"
    out_bw    = run_folder / "08_stitched_map_test_bw.png"
    out_steps = run_folder / "08_stitched_maps_steps_test"
    centers8  = run_folder / "08_centers.txt"
    gif8      = run_folder / "08_stitching_process_test.gif"

    test_stitch2(
        str(aligned) + "_orig",
        str(lidar_cart) + "_mask",
        str(out_png),
        str(out_bw),
        str(out_steps),
        str(gif8),
        str(centers8),
        pixel_threshold=150
    )

    # Step 9: combine pose file
    pose_out = run_folder / "09_pose_estimation.txt"
    combine_pose_file(
        str(centers8),
        str(angle_file),
        str(pose_out)
    )

def main():
    ensure_root()
    last_count = 0

    while True:
        cart_images = list(CART_ROOT.glob("*.jpg"))
        if len(cart_images) > last_count:
            print(f"[slam] {len(cart_images)} images found (was {last_count}) → running new SLAM run")
            try:
                run_slam()
                last_count = len(cart_images)
            except Exception as e:
                print(f"  [ERROR] SLAM pipeline failed: {e}", file=sys.stderr)
                traceback.print_exc()
                # advance so we only retry with freshly arriving images
                last_count = len(cart_images)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()

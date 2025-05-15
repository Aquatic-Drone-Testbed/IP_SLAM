import os
import time
from reconstruct_scan_single import reconstruct_scan_single

frame_folder = "scans/0_raw_spokes"
current_frame_number = 1

while True:
    expected_frame_file = f"frame{current_frame_number:05d}.txt"
    frame_path = os.path.join(frame_folder, expected_frame_file)

    if os.path.exists(frame_path):
        
        print(f"--- Frame #{current_frame_number} ---")

        # Step 1: Reconstruct scan
        reconstruct_scan_single(frame_path, "scans/1_polar_scans", "scans/1_cartesian_scans", "scans/1_heatmaps", current_frame_number)
        
        
        
        # # Step 1: Add center marker
        # cartesian_centered = add_center_marker_single(cartesian_path)

        # # Step 2: Preprocess scan
        # processed_path = preprocess_scan_single(cartesian_centered)

        # # Step 3: Estimate transformation (if not first frame)
        # if current_frame_number > 1:
        #     transformation = estimate_transformation_single(prev_processed, processed_path)
        # else:
        #     transformation = None

        # # Step 4: Orient scan
        # aligned_path = orient_scan_single(processed_path, transformation)

        # # Step 5: Update map
        # update_stitched_map(aligned_path, transformation)

        # # Step 6: Update boat path
        # update_boat_path_single(aligned_path, transformation)

        # # Step 7: Save pose info
        # save_pose_info_single(aligned_path, transformation)

        # Update for next iteration
        # prev_processed = processed_path
        
        current_frame_number += 1

    else:
        # No new frame yet — wait and check again
        time.sleep(0.1)

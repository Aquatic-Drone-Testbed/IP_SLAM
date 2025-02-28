import os
import numpy as np
import cv2
from scan_reconstruction import reconstruct_scans
from add_center_marker import add_center_marker
from preprocess_scans import preprocess_scans
from estimate_transformations import estimate_transformations
from orient_scans import orient_scans
from stitch_map import stitch_map

# os.makedirs(polar_output_folder, exist_ok=True)
# os.makedirs(cartesian_output_folder, exist_ok=True)
# os.makedirs(processed_folder, exist_ok=True)
# os.makedirs(aligned_folder, exist_ok=True)

# Step 0: Reconstruct scans
radar_data_file = "radar_data_1.txt"
polar_raw_output_folder = "scans/0_raw_polar_scans"
cartesian_raw_output_folder = "scans/0_raw_cartesian_scans"
# reconstruct_scans(radar_data_file, polar_raw_output_folder, cartesian_raw_output_folder)

# Step 1: Center Red Dot
polar_output_folder = "scans/1_polar_scans"
cartesian_output_folder = "scans/1_cartesian_scans"
add_center_marker(cartesian_raw_output_folder, cartesian_output_folder)

# Step 2: Preprocess scans
processed_folder = "scans/2_processed_scans"
preprocess_scans(cartesian_output_folder, processed_folder)

# Step 3: Estimate transformations
feature_detector = "AKAZE"
num_good_matches = 55
transformations_file = "scans/3_estimated_transformations.npy"
match_output_folder = "scans/3_feature_matches"
overlay_output_folder = "scans/3_overlay_debug"
estimate_transformations(processed_folder, transformations_file, match_output_folder, overlay_output_folder, feature_detector, num_good_matches)

# Step 4: Orient scans and create GIFs
aligned_folder = "scans/4_aligned_scans"
gif_processed_path = "scans/4_oriented_scans_processed.gif"
gif_original_path = "scans/4_oriented_scans_original.gif"
gif_unoriented_path = "scans/4_unoriented_scans.gif"
orient_scans(processed_folder, cartesian_output_folder, aligned_folder, transformations_file, gif_processed_path, gif_original_path, gif_unoriented_path)

# # Step 5: Estimate transformations for aligned
# feature_detector = "AKAZE"
# num_good_matches = 50
# transformations_file = "scans/5_estimated_transformations.npy"
# match_output_folder = "scans/5_feature_matches"
# overlay_output_folder = "scans/5_overlay_debug"
# estimate_transformations(aligned_folder + "_proc", transformations_file, match_output_folder, overlay_output_folder, feature_detector, num_good_matches)

# # Step 6: Orient scans and create GIFs
# new_aligned_folder = "scans/6_aligned_scans"
# gif_processed_path = "scans/6_oriented_scans_processed.gif"
# gif_original_path = "scans/6_oriented_scans_original.gif"
# gif_unoriented_path = "scans/6_unoriented_scans.gif"
# orient_scans(aligned_folder + "_proc", aligned_folder + "_orig", new_aligned_folder, transformations_file, gif_processed_path, gif_original_path, gif_unoriented_path)

# Step 5: Stich Images
output_map_path = "scans/5_stitched_map.png"
map_parts_output = "scans/5_stitched_maps_steps"
stitch_map(aligned_folder + "_orig", transformations_file, output_map_path, map_parts_output)
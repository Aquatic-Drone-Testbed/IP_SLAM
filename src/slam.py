import os
import numpy as np
import cv2
from scan_reconstruction import reconstruct_scans
from add_center_marker import add_center_marker
from preprocess_scans import preprocess_scans
from estimate_transformations import estimate_transformations
from orient_scans import orient_scans
from stitch_map import stitch_map
from stitch_map_pix_thresh import stitch_map_pix_thresh
from print_path import print_path
from combine_pose_file import combine_pose_file
from test_stitch import test_stitch
from test_stitch2 import test_stitch2
from cartesian_to_polar import cartesian_to_polar


# from stitch_map_de import stitch_map_de



# os.makedirs(polar_output_folder, exist_ok=True)
# os.makedirs(cartesian_output_folder, exist_ok=True)
# os.makedirs(processed_folder, exist_ok=True)
# os.makedirs(aligned_folder, exist_ok=True)

# Step 0: Reconstruct scans
radar_data_file = "../test_data/slam_radar_test_3/radar_data.txt"
polar_raw_output_folder = "scans/0_raw_polar_scans"
cartesian_raw_output_folder = "scans/0_raw_cartesian_scans"
raw_gif_output = "scans/0_raw_cartesian_scans.gif"
heatmap_output = "scans/0_heat_cartesian_scans.gif"
print(f"0...Starting Reconstruction...")
# reconstruct_scans(radar_data_file, polar_raw_output_folder, cartesian_raw_output_folder,heatmap_output) #raw_gif_output)

# Step 1: Center Red Dot
polar_output_folder = "scans/1_polar_scans"
cartesian_output_folder = "scans/1_cartesian_scans"
print(f"1...Starting Marker...")
add_center_marker(cartesian_raw_output_folder, cartesian_output_folder)

# Step 2: Preprocess scans
processed_folder = "scans/2_processed_scans"
print(f"2...Starting Preprocessing...")
preprocess_scans(cartesian_output_folder, processed_folder)

# Step 3: Estimate transformations
feature_detector = "AKAZE"
num_good_matches = 100
transformations_file = "scans/3_estimated_transformations.npy"
match_output_folder = "scans/3_feature_matches"
overlay_output_folder = "scans/3_overlay_debug"
angle_file = "scans/3_frame_angles.txt"
print(f"3...Feature Matching/Estimation...")
estimate_transformations(processed_folder, transformations_file, match_output_folder, overlay_output_folder, angle_file, feature_detector, num_good_matches)

# Step 4: Orient scans and create GIFs
aligned_folder = "scans/4_aligned_scans"
gif_processed_path = "scans/4_oriented_scans_processed.gif"
gif_original_path = "scans/4_oriented_scans_original.gif"
gif_unoriented_path = "scans/4_unoriented_scans.gif"
print(f"4...Orient Scans...")
orient_scans(processed_folder, cartesian_output_folder, aligned_folder, transformations_file, gif_processed_path, gif_original_path, gif_unoriented_path)

# # Step 5: Stitch Images
# output_map_path = "scans/5_stitched_map.png"
# map_parts_output = "scans/5_stitched_maps_steps"
# gif_map_path = "scans/5_stitching_process.gif"
# print(f"5...Stitch Map...")
# stitch_map(aligned_folder + "_orig", output_map_path, map_parts_output, gif_map_path)

# output_map_path = "scans/5_stitched_map_pt.png"
# output_map_path_bw = "scans/5_stitched_map_pt_bw.png"
# map_parts_output = "scans/5_stitched_maps_steps_pt"
# center_file = "scans/5_centers.txt"
# gif_map_path = "scans/5_stitching_process_pt.gif"
# print(f"6...Stitch Map Filtered...")
# stitch_map_pix_thresh(aligned_folder + "_orig", output_map_path, output_map_path_bw, map_parts_output, gif_map_path, center_file, pixel_threshold=150)


# # Step 6: Print Boat Path
# output_map_and_boat_path = "scans/6_stitched_map_boat_path.png"
# output_map_and_boat_gif_path = "scans/6_stitched_map_boat_path.gif"
# print_path(output_map_path, center_file, angle_file, output_map_and_boat_path, output_map_and_boat_gif_path)


reverted_polar = "scans/8_reverted_polar"
lidar_polar = "scans/8_reverted_polar_lidarized"
lidar_cartesian = "scans/8_lidar_cartesian"
cartesian_to_polar(aligned_folder + "_orig", reverted_polar, lidar_polar, lidar_cartesian)



output_map_path = "scans/8_stitched_map_test.png"
output_map_path_frames = "scans/8_stitched_map_frames"
map_parts_output = "scans/8_stitched_maps_steps_test"
center_file = "scans/8_centers.txt"
gif_map_path = "scans/8_stitching_process_test.gif"
gif_map_path_overlay = "scans/8_stitching_process_overlay.gif"
print(f"8...Stitch Map Filtered...")
# test_stitch2(lidar_cartesian, output_map_path, output_map_path_bw, map_parts_output, gif_map_path, center_file, pixel_threshold=150)
test_stitch2(aligned_folder + "_orig", lidar_cartesian + "_mask", output_map_path, output_map_path_frames, map_parts_output, gif_map_path, gif_map_path_overlay,center_file, pixel_threshold=150)


# Step 7: Combine Pose File
pose_estimation = "scans/9_pose_estimation.txt"
combine_pose_file(center_file, angle_file, pose_estimation)


# Step 6: Print Boat Path
output_map_and_boat_path = "scans/6_stitched_map_boat_path.png"
output_map_and_boat_gif_path = "scans/6_stitched_map_boat_path.gif"
print_path(output_map_path, center_file, angle_file, output_map_and_boat_path, output_map_and_boat_gif_path)


# output_map_path = "scans/5_stitched_map_de.png"
# map_parts_output = "scans/5_stitched_maps_steps_de"
# gif_map_path = "scans/5_stitching_process_de.gif"
# stitch_map_pix_thresh(aligned_folder + "_orig", output_map_path, map_parts_output, gif_map_path, 255)



import os
import numpy as np
import cv2
import re
from quantum_scan import QuantumScan

# Configurable Parameters
# ANG_VEL_THRESHOLD = 0.04         
# MAX_BAD_SPOKE_RATIO = 0.005    
ANG_VEL_THRESHOLD = 0.08    
MAX_BAD_SPOKE_RATIO = 0.005      
ANG_VEL_AXIS = 'z'    

AXIS_SELECT = {'x': 0, 'y': 1, 'z': 2}
new_frame_id = 1

def reconstruct_scans(file_path, output_folder, polar_output_folder, heatmap_output_folder):
    default_num_spokes = 250
    max_spoke_length = 256

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(polar_output_folder, exist_ok=True)
    os.makedirs(heatmap_output_folder, exist_ok=True)

    with open(file_path, "r") as f:
        lines = f.readlines()

    radar_scans = []
    velocity_maps = []
    current_scan = np.zeros((default_num_spokes, max_spoke_length), dtype=np.uint8)
    current_velocities = np.zeros(default_num_spokes, dtype=np.float32)
    received_azimuths = set()
    previous_seq_num = None
    first_azimuth = None
    dropped_spokes = 0
    bad_frames = []
    scan_id = 0
    frame_start_line = 0

    for line_num, line in enumerate(lines):
        match = re.search(r"Q_Header<\((.*?)\)>\s+Q_Data<\((.*?)\)>\s+ORIENT<geometry_msgs\.msg\.Quaternion\(x=(.*?),\s*y=(.*?),\s*z=(.*?),\s*w=(.*?)\)>\s+ANG_VEL<geometry_msgs\.msg\.Vector3\(x=(.*?),\s*y=(.*?),\s*z=(.*?)\)>\s+LIN_ACC<geometry_msgs\.msg\.Vector3\(x=(.*?),\s*y=(.*?),\s*z=(.*?)\)>\s+COMP<(\d+)>\s+", line)
        if match:
            header_values = tuple(map(int, match.group(1).split(", ")))
            data_values = tuple(map(int, match.group(2).split(", ")))
            ang_vel = tuple(map(float, (match.group(7), match.group(8), match.group(9))))
            ang_vel_value = ang_vel[AXIS_SELECT[ANG_VEL_AXIS]]

            qs = QuantumScan(*header_values[:9], data_values)

            if first_azimuth is None:
                first_azimuth = qs.azimuth

            if previous_seq_num is not None and qs.seq_num > previous_seq_num + 1:
                dropped_spokes += qs.seq_num - previous_seq_num - 1
            previous_seq_num = qs.seq_num

            azimuth_index = (qs.azimuth + default_num_spokes // 2) % default_num_spokes

            if azimuth_index == first_azimuth and len(received_azimuths) > 0:
                bad_spokes = np.sum(np.abs(current_velocities) > ANG_VEL_THRESHOLD)
                bad_ratio = bad_spokes / default_num_spokes

                if bad_ratio <= MAX_BAD_SPOKE_RATIO:
                    # print(f"Frame #{scan_id} (lines {frame_start_line}-{line_num}): PASS")
                    radar_scans.append(current_scan.copy())
                    velocity_maps.append(current_velocities.copy())
                else:
                    # print(f"Frame #{scan_id} (lines {frame_start_line}-{line_num}): FAIL - {bad_ratio:.2%} bad spokes")
                    bad_frames.append(scan_id)

                scan_id += 1
                frame_start_line = line_num + 1
                current_scan.fill(0)
                current_velocities.fill(0)
                received_azimuths.clear()

            current_scan[azimuth_index, :len(qs.data)] = qs.data
            current_velocities[azimuth_index] = ang_vel_value
            received_azimuths.add(azimuth_index)

    total_scans = len(radar_scans)
    pad_width = len(str(total_scans))

    for scan_id, (scan, vel_map) in enumerate(zip(radar_scans, velocity_maps)):
        filename_suffix = str(scan_id).zfill(pad_width)

        polar_path = os.path.join(output_folder, f"polar_scan_{filename_suffix}.jpg")
        cv2.imwrite(polar_path, (scan / np.max(scan) * 255).astype(np.uint8))

        radar_image = cv2.warpPolar(
            src=scan,
            dsize=(2 * max_spoke_length, 2 * max_spoke_length),
            center=(max_spoke_length, max_spoke_length),
            maxRadius=max_spoke_length,
            flags=cv2.WARP_INVERSE_MAP
        )
        radar_image = cv2.rotate(radar_image, cv2.ROTATE_90_CLOCKWISE)

        mask = np.zeros_like(radar_image, dtype=np.uint8)
        cv2.circle(mask, (max_spoke_length, max_spoke_length), max_spoke_length - 10, (255, 255, 255), thickness=-1)
        radar_image = cv2.bitwise_and(radar_image, radar_image, mask=mask)

        cartesian_path = os.path.join(polar_output_folder, f"cartesian_scan_{filename_suffix}.jpg")
        cv2.imwrite(cartesian_path, radar_image)

        heatmap_color = np.zeros((default_num_spokes, max_spoke_length, 3), dtype=np.uint8)

        norm_vel = np.clip((np.abs(vel_map) / ANG_VEL_THRESHOLD) * 255, 0, 255).astype(np.uint8)
        for i in range(default_num_spokes):
            if np.any(scan[i, :] > 0):
                spoke_color = cv2.applyColorMap(np.full((1, 1), norm_vel[i], dtype=np.uint8), cv2.COLORMAP_JET)[0, 0]
                non_zero_mask = scan[i, :] > 0
                heatmap_color[i, non_zero_mask] = spoke_color

        heatmap_cartesian = cv2.warpPolar(src=heatmap_color, dsize=(2 * max_spoke_length, 2 * max_spoke_length), center=(max_spoke_length, max_spoke_length), maxRadius=max_spoke_length, flags=cv2.WARP_INVERSE_MAP)
        heatmap_cartesian = cv2.rotate(heatmap_cartesian, cv2.ROTATE_90_CLOCKWISE)
        heatmap_cartesian = cv2.bitwise_and(heatmap_cartesian, heatmap_cartesian, mask=mask)

        heatmap_path = os.path.join(heatmap_output_folder, f"heatmap_scan_{filename_suffix}.jpg")
        cv2.imwrite(heatmap_path, heatmap_cartesian)

    # print(f"\nFailed Frames: {bad_frames}")

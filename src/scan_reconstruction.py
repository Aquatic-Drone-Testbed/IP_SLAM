import os
import numpy as np
import cv2
import re
from quantum_scan import QuantumScan

def reconstruct_scans(file_path, output_folder, polar_output_folder):
    default_num_spokes = 250
    max_spoke_length = 256

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(polar_output_folder, exist_ok=True)

    with open(file_path, "r") as f:
        lines = f.readlines()

    radar_scans = []
    current_scan = np.zeros((default_num_spokes, max_spoke_length), dtype=np.uint8)
    received_azimuths = set()
    previous_seq_num = None
    first_azimuth = None
    dropped_spokes = 0

    for line in lines:
        match = re.search(r"Q_Header<\((.*?)\)>\s+Q_Data<\((.*?)\)>", line)
        if match:
            header_values = tuple(map(int, match.group(1).split(", ")))
            data_values = tuple(map(int, match.group(2).split(", ")))

            qs = QuantumScan(*header_values[:9], data_values)

            if first_azimuth is None:
                first_azimuth = qs.azimuth

            if previous_seq_num is not None and qs.seq_num > previous_seq_num + 1:
                dropped_spokes += qs.seq_num - previous_seq_num - 1
            previous_seq_num = qs.seq_num

            azimuth_index = (qs.azimuth + default_num_spokes // 2) % default_num_spokes

            if azimuth_index == first_azimuth and len(received_azimuths) > 0:
                radar_scans.append(current_scan.copy())
                current_scan.fill(0)
                received_azimuths.clear()

            current_scan[azimuth_index, :len(qs.data)] = qs.data
            received_azimuths.add(azimuth_index)

    if len(received_azimuths) > 0:
        radar_scans.append(current_scan.copy())

    for scan_id, scan in enumerate(radar_scans):
        cartesian_path = os.path.join(output_folder, f"polar_scan_{scan_id}.jpg")
        cv2.imwrite(cartesian_path, (scan / np.max(scan) * 255).astype(np.uint8))

        radar_image = cv2.warpPolar(
            src=scan, dsize=(2 * max_spoke_length, 2 * max_spoke_length), 
            center=(max_spoke_length, max_spoke_length), maxRadius=max_spoke_length, 
            flags=cv2.WARP_INVERSE_MAP
        )
        radar_image = cv2.rotate(radar_image, cv2.ROTATE_90_CLOCKWISE)

        mask = np.zeros_like(radar_image, dtype=np.uint8)
        cv2.circle(mask, (max_spoke_length, max_spoke_length), max_spoke_length - 10, (255, 255, 255), thickness=-1)
        radar_image = cv2.bitwise_and(radar_image, radar_image, mask=mask)

        polar_path = os.path.join(polar_output_folder, f"cartesian_scan_{scan_id}.jpg")
        cv2.imwrite(polar_path, radar_image)


import os
import re
import time

def split_radar_frames(input_file, output_folder, save_delay=2.5):
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'r') as f:
        lines = f.readlines()

    azimuth_regex = re.compile(r"Q_Header<\((.*?)\)>")

    frame_lines = []
    frame_id = 1
    prev_azimuth = None

    for line in lines:
        match = azimuth_regex.search(line)
        if not match:
            continue

        header_values = list(map(int, match.group(1).split(', ')))
        azimuth = header_values[7]

        if prev_azimuth is not None and azimuth < prev_azimuth:
            filename = f"frame{frame_id:05d}.txt"
            with open(os.path.join(output_folder, filename), 'w') as f_out:
                f_out.writelines(frame_lines)

            time.sleep(save_delay)

            frame_id += 1
            frame_lines = []

        frame_lines.append(line)
        prev_azimuth = azimuth

    if frame_lines:
        filename = f"frame{frame_id:05d}.txt"
        with open(os.path.join(output_folder, filename), 'w') as f_out:
            f_out.writelines(frame_lines)


split_radar_frames("../test_data/slam_radar_test_3/radar_data.txt", "scans/raw_dumps", save_delay=0)

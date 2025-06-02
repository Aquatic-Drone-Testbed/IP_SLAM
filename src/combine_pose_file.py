import os
def combine_pose_file(position_file, angle_file, output_file, current_pose):
    latest_pose = None
    if os.path.exists(position_file) and os.path.exists(angle_file):
        with open(position_file, 'r') as pos_f, open(angle_file, 'r') as ang_f, open(output_file, 'w') as out_f:
                # Read all lines from the angle file to extract frame numbers
                angle_lines = ang_f.readlines()

                # Iterate through the position file
                for frame_number, pos_line in enumerate(pos_f, 1):
                    pos_parts = pos_line.strip().split(' ')
                    ang_parts = angle_lines[frame_number - 1].strip().split(' ')

                    x, y = pos_parts
                    frame_ang, angle = ang_parts
                    if(frame_ang == "n"): continue

                    position_str = f"({x},{y})"
                    latest_pose = f"{frame_number}\t{position_str}\t{angle}\n"
                    out_f.write(latest_pose)

                if latest_pose:
                    with open(current_pose, 'w') as current_pose_f:
                        current_pose_f.write(latest_pose)

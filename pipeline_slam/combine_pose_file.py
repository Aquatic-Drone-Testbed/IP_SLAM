def combine_pose_file(position_file, angle_file, output_file):
   with open(position_file, 'r') as pos_f, open(angle_file, 'r') as ang_f, open(output_file, 'w') as out_f:
        # Read all lines from the angle file to extract frame numbers
        angle_lines = ang_f.readlines()

        # Iterate through the position file
        for frame_number, pos_line in enumerate(pos_f, 1):
            pos_parts = pos_line.strip().split(' ')
            ang_parts = angle_lines[frame_number - 1].strip().split(' ')

            x, y = pos_parts
            frame_ang, angle = ang_parts

            position_str = f"({x},{y})"
            out_f.write(f"{frame_number}\t{position_str}\t{angle}\n")

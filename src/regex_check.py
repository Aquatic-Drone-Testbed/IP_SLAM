import re

pattern = (
    r"\[\d+\.\d+\]\t"  # Match and ignore the timestamp (e.g., [1740776225.6419556])
    r"Q_Header<\((.*?)\)>\t"
    r"Q_Data<\((.*?)\)>\t"
    r"ORIENT<geometry_msgs.msg.Quaternion\(x=(-?\d+\.\d+), y=(-?\d+\.\d+), z=(-?\d+\.\d+), w=(-?\d+\.\d+)\)>\t"
    r"ANG_VEL<geometry_msgs.msg.Vector3\(x=(-?\d+\.\d+), y=(-?\d+\.\d+), z=(-?\d+\.\d+)\)>\t"
    r"LIN_ACC<geometry_msgs.msg.Vector3\(x=(-?\d+\.\d+), y=(-?\d+\.\d+), z=(-?\d+\.\d+)\)>\t"
    r"COMP<(\d+)>"
)

with open("../test_data/slam_radar_test_3/radar_data.txt", "r") as f:
    for line in f:
        line = line.strip()  # Remove leading/trailing whitespace
        if not line:  # Skip empty lines
            continue

        match = re.search(r"\[(\d+\.\d+)\]\s+Q_Header<\((.*?)\)>\s+Q_Data<\((.*?)\)>\s+ORIENT<geometry_msgs\.msg\.Quaternion\(x=(.*?),\s*y=(.*?),\s*z=(.*?),\s*w=(.*?)\)>\s+ANG_VEL<geometry_msgs\.msg\.Vector3\(x=(.*?),\s*y=(.*?),\s*z=(.*?)\)>\s+LIN_ACC<geometry_msgs\.msg\.Vector3\(x=(.*?),\s*y=(.*?),\s*z=(.*?)\)>\s+COMP<(\d+)>"
, line)
        if match:
            print("Matched Header:", match.group(1))
            print("Matched Data:", match.group(2))
            print("Quaternion:", match.group(3), match.group(4), match.group(5), match.group(6))
            print("Angular Velocity:", match.group(7), match.group(8), match.group(9))
            print("Linear Acceleration:", match.group(10), match.group(11), match.group(12))
            print("COMP:", match.group(13))
        else:
            print("No match for:", repr(line))  # Debugging output


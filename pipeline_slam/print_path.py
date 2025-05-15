from PIL import Image, ImageDraw
import math

def print_path(map_image_path, center_txt_path, angles_txt_path, output_path_img, output_path_gif, line_length=5):
    
    image = Image.open(map_image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size


    with open(center_txt_path, "r") as f_centers, open(angles_txt_path, "r") as f_angles:
        centers = [tuple(map(int, line.strip().split())) for line in f_centers]
        angles = [float(line.strip().split()[1]) for line in f_angles]  

    gif_frames = []

    for (x, y), angle_deg in zip(centers, angles):
        image = Image.open(map_image_path)
        draw = ImageDraw.Draw(image)


        angle_rad = math.radians(angle_deg + 90)
        x_end = int(x + line_length * math.cos(angle_rad))
        y_end = int(y + line_length * math.sin(angle_rad))
        draw.line((x, y, x_end, y_end), fill="green", width=1)

        draw.point((x, y), fill="red")

        gif_frame = image.convert("RGB")
        gif_frames.append(gif_frame.copy())

    image.save(output_path_img)

    gif_frames[0].save(output_path_gif, save_all=True, append_images=gif_frames[1:], loop=0, duration=200)



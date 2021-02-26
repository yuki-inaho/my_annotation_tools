import cv2
import click
import os
import cvui
import numpy as np
from pathlib import Path
from scripts.utils import get_image_pathes

HOME_PATH = str(Path(os.environ["HOME"]).resolve())


def get_image(i, image_pathes):
    return cv2.imread(image_pathes[i]), Path(image_pathes[i]).name


def resize_material_to_show(image, image_size_to_show):
    half_image_size_to_show = int(image_size_to_show / 2)
    height, width, _ = image.shape
    image_to_show = np.zeros((image_size_to_show, image_size_to_show, 3), dtype=np.uint8)

    scale = image_size_to_show / max(height, width)
    height_scaled = int(np.round(height * scale / 2).astype(np.int32)) * 2
    width_scaled = int(np.round(width * scale / 2).astype(np.int32)) * 2

    image_resized = cv2.resize(image, (width_scaled, height_scaled))
    start_y = half_image_size_to_show - int(height_scaled / 2)
    end_y = half_image_size_to_show + int(height_scaled / 2)
    start_x = half_image_size_to_show - int(width_scaled / 2)
    end_x = half_image_size_to_show + int(width_scaled / 2)
    image_to_show[start_y:end_y, start_x:end_x, :] = image_resized
    return image_to_show


@click.command()
@click.option("--input-image-dir", "-i", default=f"{HOME_PATH}/data/yokojiro2/rgb_15-19_output")
@click.option("--output-txt-path", "-o", default=f"{HOME_PATH}/data/yokojiro2/extracted_images.txt")
def main(input_image_dir, output_txt_path):
    image_pathes = get_image_pathes(Path(input_image_dir))
    n_image_indices = len(image_pathes)
    assert n_image_indices > 0

    image_size_to_show = 768
    frame = np.zeros((900, 1400, 3), dtype=np.uint8)
    image_loc_start = 50
    image_loc_end = image_loc_start + image_size_to_show

    selected_images = []

    current_image_index = 0
    window_name = "Material Selection"
    cvui.init(window_name)
    while True:
        # Fill the frame with a nice color
        frame[:] = (49, 52, 49)

        # Show image and its name
        image, image_name = get_image(current_image_index, image_pathes)
        image_resized = resize_material_to_show(image, image_size_to_show)
        frame[image_loc_start:image_loc_end, image_loc_start:image_loc_end, :] = image_resized
        cvui.printf(frame, 100, image_loc_end + 20, 0.6, 0x00FF00, "image_name ({}/{}): {}".format(current_image_index, n_image_indices, image_name))

        # Check
        target_to_use = image_name in selected_images

        cvui.printf(frame, image_loc_end + 50, 100, 0.8, 0x00FF00, "Is target image to use? :")
        if target_to_use:
            cvui.printf(frame, image_loc_end + 500, 100, 2.0, 0xFF8800, "@")
        else:
            cvui.printf(frame, image_loc_end + 500, 100, 2.0, 0x0088FF, "X")

        if cvui.button(frame, image_loc_end + 200, 300, 300, 100, "Dump"):
            if len(selected_images) > 0:
                np.savetxt(output_txt_path, selected_images, fmt="%s", delimiter=",")

        cvui.printf(frame, image_loc_end + 100, 600, 0.8, 0x00FF00, "Number of Selected Images:{}".format(len(selected_images)))

        cvui.update()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20)

        if key & 0xFF == ord("l"):
            if current_image_index < n_image_indices - 1:
                current_image_index += 1
        elif key & 0xFF == ord("j"):
            if current_image_index > 0:
                current_image_index -= 1
        elif key & 0xFF == ord("o"):
            if image_name in selected_images:
                selected_images.remove(image_name)
            else:
                selected_images.append(image_name)                
        if key == 27:
            break


if __name__ == "__main__":
    main()

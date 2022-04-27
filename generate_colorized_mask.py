
import click
from pathlib import Path
import cv2
import numpy as np
import shutil

SCRIPT_DIR = str(Path(__file__).parent.resolve())

dict_idx2color = {
    0: (0, 0, 0),
    1: (110, 110, 255),  # Oyagi
    2: (150, 249, 152),  # Aspara
    3: (255, 217, 81),  # Ground
    4: (252, 51, 255),  # Tube
    5: (84, 223, 255),  # Pole
    6: (128, 0, 255)  # Protector
}


def colorize_mask(mask_image, n_label):
    mask_colorized = np.zeros(
        [mask_image.shape[0], mask_image.shape[1], 3], dtype=np.uint8
    )
    for l in range(n_label+1) :
        mask_indices_lth_label = np.where(mask_image == l)
        mask_colorized[mask_indices_lth_label[0], mask_indices_lth_label[1], :] = dict_idx2color[l]
    return mask_colorized


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-mask-dir", "-o", default=f"{SCRIPT_DIR}/colorized")
def main(input_data_dir, output_mask_dir):
    image_path_list = [str(path) for path in Path(input_data_dir).glob("*.png")]

    # Generate mask images
    output_mask_dir_path = Path(output_mask_dir)
    if output_mask_dir_path.exists():
        shutil.rmtree(output_mask_dir_path)
    output_mask_dir_path.mkdir()

    for image_path in image_path_list:
        image_name = Path(image_path).name
        mask = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        mask_colorized = colorize_mask(mask, 5)
        output_path = str(Path(output_mask_dir).joinpath(image_name))
        cv2.imwrite(output_path, mask_colorized)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()

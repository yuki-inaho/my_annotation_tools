import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
from scripts.lens_undistortion import LensUndistorter

import pdb


SCRIPT_DIR = str(Path(__file__).parent)

def get_input_image_path_list(input_image_dir_path):
    extf = [".jpg", ".png"]
    input_image_path_list = [str(path) for path in input_image_dir_path.glob("*") if path.suffix in extf]
    return np.sort(input_image_path_list)


@click.command()
@click.option("--config-file-path", "-c", default=f"{SCRIPT_DIR}/cfg/dualzense_out.toml")
@click.option("--input-image-dir", "-i", default=f"{SCRIPT_DIR}/data/raw")
@click.option("--output-image-dir", "-o", default=f"{SCRIPT_DIR}/data/undistorted")
def main(config_file_path, input_image_dir, output_image_dir):
    # Define undistorter
    undistorter = LensUndistorter(config_file_path)
    output_image_dir_path = Path(output_image_dir)
    if not output_image_dir_path.exists():
        output_image_dir_path.mkdir()

    # Get Images
    input_image_dir_path = Path(input_image_dir)
    input_image_path_list = get_input_image_path_list(input_image_dir_path)

    for input_image_path in tqdm(input_image_path_list):
        image_raw = cv2.imread(input_image_path)
        image_undistorted = undistorter.correction(image_raw)
        base_name = Path(input_image_path).name
        cv2.imwrite(str(Path(output_image_dir_path, base_name)), image_undistorted)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
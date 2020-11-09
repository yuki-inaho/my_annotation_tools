import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
from scripts.lens_undistortion import LensUndistorter

import pdb


SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--config-file-path", "-c", default=f"{SCRIPT_DIR}/cfg/dualzense_out.toml")
@click.option("--input-image-dir", "-c", default=f"{SCRIPT_DIR}/data/raw")
@click.option("--output-image-dir", "-c", default=f"{SCRIPT_DIR}/data/undistorted")
def main(config_file_path, input_image_dir, output_image_dir):
    # Define undistorter
    undistorter = LensUndistorter(config_file_path)
    output_image_dir_path = Path(output_image_dir)
    if not output_image_dir_path.exists():
        output_image_dir_path.mkdir()

    # Get Images
    input_image_dir_path = Path(input_image_dir)
    input_image_path_list = np.sort([str(image_path) for image_path in input_image_dir_path.glob("*.png")])

    for input_image_path in tqdm(input_image_path_list):
        image_raw = cv2.imread(input_image_path)
        image_undistorted = undistorter.correction(image_raw)
        base_name = Path(input_image_path).name
        cv2.imwrite(str(Path(output_image_dir_path, base_name)), image_undistorted)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
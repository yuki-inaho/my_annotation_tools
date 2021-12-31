from os import mkdir
from sys import flags
import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
import pdb
import shutil
from scripts.utils import mkdir_from_path


SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-image-dir", "-i", default=f"{SCRIPT_DIR}/data/image_raw")
@click.option("--output-image-dir", "-o", default=f"{SCRIPT_DIR}/data/image_resized")
@click.option("--resize-rate", "-r", default=1.0 / 3)
def main(input_image_dir, output_image_dir, resize_rate):
    image_path_list = [str(path) for path in Path(input_image_dir).glob("*") if path.suffix in [".png", ".jpg"]]
    mkdir_from_path(output_image_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        from_image_path = image_path
        to_image_path = str(Path(output_image_dir, base_name))
        rgb_image = cv2.imread(from_image_path)
        rgb_image_resized = cv2.resize(rgb_image, None, fx=resize_rate, fy=resize_rate)
        cv2.imwrite(to_image_path, rgb_image_resized)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()

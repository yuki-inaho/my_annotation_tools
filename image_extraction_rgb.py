from os import mkdir
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
@click.option("--input-image-dir", "-i", default=f"{SCRIPT_DIR}/data/image_to_extraction")
@click.option("--output-image-dir", "-o", default=f"{SCRIPT_DIR}/data/image_extracted")
def main(input_image_dir, output_image_dir):
    image_pathes = Path(input_image_dir).glob("*_rgb.png")
    image_path_list = [str(image_path) for image_path in image_pathes]
    mkdir_from_path(output_image_dir)

    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        from_image_path = Path(input_image_dir, base_name)
        to_image_path = Path(output_image_dir, base_name)
        shutil.copy(from_image_path, to_image_path)

if __name__ == "__main__":

    main()


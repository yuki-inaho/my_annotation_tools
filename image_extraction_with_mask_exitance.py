import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
import pdb
import shutil


SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-image-dir", "-c", default=f"{SCRIPT_DIR}/data/image_to_extraction")
@click.option("--output-image-dir", "-c", default=f"{SCRIPT_DIR}/data/image_extracted")
@click.option("--mask-dir", "-c", default=f"{SCRIPT_DIR}/data/mask")
def main(input_image_dir, output_image_dir, mask_dir):
    mask_pathes = Path(mask_dir).glob("*.png")
    mask_path_list = [str(mask_path) for mask_path in mask_pathes]

    for mask_path in tqdm(mask_path_list):
        base_name = Path(mask_path).name
        from_image_path = Path(input_image_dir, base_name)
        to_image_path = Path(output_image_dir, base_name)
        shutil.copy(from_image_path, to_image_path)

if __name__ == "__main__":

    main()


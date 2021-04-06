import os
import cv2
import click
import numpy as np
from pathlib import Path
from tqdm import tqdm

SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-image-dir", "-i", default=f"{SCRIPT_DIR}/data/Image_raw")
@click.option("--output-image-dir", "-o", default=f"{SCRIPT_DIR}/data/Image")
@click.option("--flag-mkdir", "-m", is_flag=True)
def main(input_image_dir, output_image_dir, flag_mkdir):
    if not flag_mkdir:  ##i.e. input_image_directory == output_image_directory
        output_image_dir = input_image_dir
        output_image_dir_path = Path(input_image_dir)
    else:
        output_image_dir_path = Path(output_image_dir)
        if not output_image_dir_path.exists():
            output_image_dir_path.mkdir()

    extf = [".jpg", ".png"]
    image_pathes = [path for path in Path(input_image_dir).glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]

    for image_path in tqdm(image_path_list):
        image = cv2.imread(image_path)
        base_name = Path(image_path).name
        base_name = base_name.replace(".png", ".jpg")
        output_image_path = str(Path(output_image_dir, base_name))
        cv2.imwrite(output_image_path, image)
        cv2.waitKey(10)
        if (not flag_mkdir) and (Path(image_path).suffix == ".png"):
            os.remove(image_path)

if __name__ == "__main__":
    main()

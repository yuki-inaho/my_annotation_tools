import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
import pdb


SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-image-dir", "-i", default=f"{SCRIPT_DIR}/data/Image_raw")
@click.option("--output-image-dir", "-o", default=f"{SCRIPT_DIR}/data/Image")
def main(input_image_dir, output_image_dir):
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
        cv2.imwrite(str(Path(output_image_dir, base_name)), image)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()

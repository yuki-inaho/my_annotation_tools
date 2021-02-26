import os
import click
from pathlib import Path
from tqdm import tqdm
from scripts.utils import mkdir_from_path
import shutil
import pdb

SCRIPT_DIR = str(Path(__file__).parent)
HOME_PATH = str(Path(os.environ["HOME"]).resolve())


def get_image_names_on_white_list(input_txt_path):
    with open(input_txt_path, "r") as f:
        image_name_white_list = f.readlines()
    image_name_white_list = [name_str.replace("\n", "") for name_str in image_name_white_list]
    return image_name_white_list


@click.command()
@click.option("--input-image-dir", "-i", default=f"{HOME_PATH}/data/yokojiro2/rgb_15-19_jpg")
@click.option("--input-txt-path", "-t", default=f"{HOME_PATH}/data/yokojiro2/extracted_images.txt")
@click.option("--output-image-dir", "-o", default=f"{HOME_PATH}/data/yokojiro2/extracted")
def main(input_image_dir, input_txt_path, output_image_dir):
    mkdir_from_path(output_image_dir)
    image_names = get_image_names_on_white_list(input_txt_path)
    input_image_dir_pathlib = Path(input_image_dir)
    output_image_dir_pathlib = Path(output_image_dir)

    for image_name in tqdm(image_names):
        input_image_path = str(input_image_dir_pathlib.joinpath(image_name))
        output_image_path = str(output_image_dir_pathlib.joinpath(image_name))
        shutil.copy(input_image_path, output_image_path)


if __name__ == "__main__":

    main()

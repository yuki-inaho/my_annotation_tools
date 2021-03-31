import click
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def get_image_pathes(input_dir_pathlib: Path):
    extf = [".jpg", ".png"]
    image_pathes = [path for path in input_dir_pathlib.glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


@click.command()
@click.option("--source-image-dir", "-s", default="Image_whole")
@click.option("--removal-target-dir", "-t", default="Image_partial")
@click.option("--output-data-dir", "-o", default="Image")
def main(source_image_dir, removal_target_dir, output_data_dir):
    output_data_dir_pathlib = Path(output_data_dir)
    output_data_dir_pathlib.mkdir(exist_ok=True)

    whole_data_dir_pathlib = Path(source_image_dir)
    partial_data_dir_pathlib = Path(removal_target_dir)

    image_list_partial = get_image_pathes(partial_data_dir_pathlib)
    image_list_whole = get_image_pathes(whole_data_dir_pathlib)

    image_name_list_partial = [Path(image_path).name for image_path in image_list_partial]
    image_name_list_whole = [Path(image_path).name for image_path in image_list_whole]

    intersect_images_name_list = np.setdiff1d(image_name_list_whole, image_name_list_partial)
    for image_name in tqdm(intersect_images_name_list):
        input_image_path = str(whole_data_dir_pathlib.joinpath(image_name))
        image = cv2.imread(input_image_path)
        output_image_name = Path(image_name).stem + ".jpg"
        output_image_path = str(output_data_dir_pathlib.joinpath(output_image_name))
        cv2.imwrite(output_image_path, image)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()
import cv2
import os
import click
import shutil
from pathlib import Path
from tqdm import tqdm
from scripts.utils import get_image_pathes, load_json, dump_json


HOME_PATH = os.environ["HOME"]


@click.command()
@click.option("--input-dir-path", "-i", default=f"{HOME_PATH}/superannotate_preupload/converted")
@click.option("--output-dir-path", "-o", default=f"{HOME_PATH}/superannotate_preupload/vector")
def main(input_dir_path, output_dir_path):
    # Set input 
    input_dir_pathlib = Path(input_dir_path)
    output_dir_pathlib = Path(output_dir_path)
    if output_dir_pathlib.exists():
        shutil.rmtree(output_dir_path)
    output_dir_pathlib.mkdir()

    # Get Image pathes from input image directory
    input_image_path_list = get_image_pathes(input_dir_pathlib)
    for input_image_path in tqdm(input_image_path_list):
        image = cv2.imread(input_image_path)
        image_height, image_width, _ = image.shape

        # Get image & annotation path info
        image_name = Path(input_image_path).name
        image_ext = Path(input_image_path).suffix
        annotation_input_json_name = image_name + ".json"
        annotation_input_json_path = str(Path(input_dir_path, annotation_input_json_name))
        annotation_data = load_json(annotation_input_json_path)
        annotation_data["metadata"]["width"] = image_width
        annotation_data["metadata"]["height"] = image_height

        # Dump image and annotation data
        output_image_path = str(Path(output_dir_path, image_name))
        output_annotation_path = str(Path(output_dir_path, image_name + "___objects.json"))

        shutil.copy(input_image_path, output_image_path)
        dump_json(output_annotation_path, annotation_data)


if __name__ == "__main__":
    main()
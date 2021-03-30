import os
import click
import shutil
from pathlib import Path
from tqdm import tqdm
from scripts.utils import get_image_pathes, mkdir_from_path


HOME_PATH = os.environ["HOME"]


@click.command()
@click.option("--input-dir-path", "-i", default=f"{HOME_PATH}/superannotate_preupload/sa_desktop_modified")
@click.option("--output-dir-path", "-o", default=f"")
def main(input_dir_path, output_dir_path):
    # Set input
    input_dir_pathlib = Path(input_dir_path)
    is_output_same_dir = output_dir_path == ""
    if not is_output_same_dir:
        mkdir_from_path(output_dir_path)

    # Get Image pathes from input image directory
    input_image_path_list = get_image_pathes(input_dir_pathlib)
    for input_image_path in tqdm(input_image_path_list):
        # Get image & annotation path info
        image_name = Path(input_image_path).name
        annotation_input_json_name = image_name + ".json"
        annotation_input_json_path = str(Path(input_dir_path, annotation_input_json_name))

        # Dump image and annotation data
        if not is_output_same_dir:
            output_image_path = str(Path(output_dir_path, image_name))
            output_annotation_path = str(Path(output_dir_path, image_name + "___objects.json"))
            shutil.copy(input_image_path, output_image_path)
            shutil.copy(annotation_input_json_path, output_annotation_path)
        else:
            output_annotation_path = str(Path(input_dir_path, image_name + "___objects.json"))
            shutil.move(annotation_input_json_path, output_annotation_path)


if __name__ == "__main__":
    main()
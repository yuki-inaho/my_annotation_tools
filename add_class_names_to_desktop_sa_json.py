import os
import click
import shutil
from pathlib import Path

from tqdm import tqdm
from scripts.utils import read_json, write_json, get_image_pathes, mkdir_from_path
from scripts.category import AnnotationClassManager


HOME_PATH = os.environ["HOME"]


@click.command()
@click.option("--input-dir-path", "-i", default=f"{HOME_PATH}/superannotate_preupload/sa_desktop")
@click.option("--output-dir-path", "-o", default=f"{HOME_PATH}/superannotate_preupload/sa_desktop_modified")
@click.option("--classes-json", "-c", default=f"cfg/classes.json")
@click.option("--flag-mkdir", "-m", is_flag=True)
def main(input_dir_path, output_dir_path, classes_json, flag_mkdir):
    # Set input
    input_dir_pathlib = Path(input_dir_path)
    annotation_class_manager = AnnotationClassManager(classes_json)
    if flag_mkdir:
        mkdir_from_path(output_dir_path)

    # Get Image pathes from input image directory
    input_image_path_list = get_image_pathes(input_dir_pathlib)
    for input_image_path in tqdm(input_image_path_list):
        # Get image & annotation path info
        image_name = Path(input_image_path).name
        annotation_input_json_name = image_name + ".json"
        annotation_input_json_path = str(Path(input_dir_path, annotation_input_json_name))
        annotation_json = read_json(annotation_input_json_path)
        n_instances = len(annotation_json["instances"])
        for i in range(n_instances):
            instance_info_ith = annotation_json["instances"][i]
            instance_info_ith["className"] = annotation_class_manager.id2name(instance_info_ith["classId"])
            annotation_json["instances"][i] = instance_info_ith

        # Dump image and annotation data
        output_image_path = str(Path(output_dir_path, image_name))
        output_annotation_path = str(Path(output_dir_path, annotation_input_json_name))

        if flag_mkdir:
            shutil.copy(input_image_path, output_image_path)
            write_json(output_annotation_path, annotation_json)
        else:
            write_json(annotation_input_json_path, annotation_json)


if __name__ == "__main__":
    main()
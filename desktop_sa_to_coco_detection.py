import os
import click
import shutil
import json
from pathlib import Path


import numpy as np
from typing import List
from tqdm import tqdm
from enum import Enum
from scripts.annotation import BoundingBox, ImageSize, COCOInstanceAnnotation
from scripts.utils import load_json, dump_json, get_image_pathes
from datetime import datetime
import pdb


HOME_PATH = os.environ["HOME"]


class Label(Enum):
    Asparagus = 1


def extract_and_parse_sa_annotated_instance(annotation_sa_data, image_size: ImageSize) -> List[BoundingBox]:
    instance_sa_list = annotation_sa_data["instances"]
    bb_list = []
    for instance_sa in instance_sa_list:
        bb_obj = BoundingBox(image_size)
        x_min = instance_sa["points"]["x1"]
        y_min = instance_sa["points"]["y1"]
        x_max = instance_sa["points"]["x2"]
        y_max = instance_sa["points"]["y2"]
        bb_obj.set_bounding_box_xyxy(x_min, y_min, x_max, y_max)
        bb_obj.set_category_id(instance_sa["classId"])
        bb_list.append(bb_obj)
    return bb_list


class COCOAnnotationInfo:
    def __init__(self):

    def _set_date_info(self):
        datetime.now()
        self._year = 2021,
        "date_created": "25/02/2021"

@click.command()
@click.option("--input-dir-path", "-i", default=f"{HOME_PATH}/data/aspara_tip_small")
@click.option("--output-dir-path", "-o", default=f"{HOME_PATH}/data/aspara_tip_coco")
@click.option("--image-width", "-w", type=int, default=1280)
@click.option("--image-height", "-h", type=int, default=720)
def main(input_dir_path, output_dir_path, image_width, image_height):
    # Get input and output directory path, and generate output directory
    input_dir_pathlib = Path(input_dir_path)
    output_dir_pathlib = Path(output_dir_path)
    if output_dir_pathlib.exists():
        shutil.rmtree(output_dir_path)
    output_dir_pathlib.mkdir()

    input_image_size = ImageSize(image_width, image_height)  # assume all images have the same image size
    input_image_path_list = get_image_pathes(input_dir_pathlib)
    current_annotation_data_count = 0

    for image_index, input_image_path in enumerate(tqdm(input_image_path_list)):
        # Get image & annotation data path
        image_name = Path(input_image_path).name
        annotation_json_name = image_name + ".json"
        annotation_json_path = str(Path(input_dir_path, annotation_json_name))

        # Load annotation json
        annotation_sa_data = load_json(annotation_json_path)
        bb_object_list = extract_and_parse_sa_annotated_instance(annotation_sa_data, input_image_size)
        n_bounding_box = len(bb_object_list)

        annotation_id_list = np.arange(n_bounding_box) + current_annotation_data_count
        coco_instances_info = COCOInstanceAnnotation(image_index, image_name, input_image_size, annotation_id_list, bb_object_list)
        current_annotation_data_count += n_bounding_box


        # @
        output_image_path = str(Path(output_dir_path, image_name))
        shutil.copy(input_image_path, output_image_path)


if __name__ == "__main__":
    main()
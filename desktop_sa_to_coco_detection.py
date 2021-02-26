import os
import click
import shutil
import json
from pathlib import Path

from typing import List
from tqdm import tqdm
from enum import IntEnum
from scripts.annotation import BoundingBox, ImageSize
from scripts.utils import load_json, dump_json, get_image_pathes


HOME_PATH = os.environ["HOME"]


class Label(IntEnum):
    Tip = 0
    Whole = 1


# TODO: move to scripts/annotation.py script
class COCOInstanceAnnotation:
    def __init__(
        self,
        image_id: int,
        image_name: str,
        image_size: ImageSize,
        annotation_id_list: List[int],
        class_id_list: List[int],
        bb_obj_list: List[BoundingBox],
        licence_id: int = 1,
    ):
        self._image_id = image_id
        self._image_name = image_name
        self._image_size = image_size
        self._annotation_id_list = annotation_id_list
        self._class_id_list = class_id_list
        self._bb_obj_list = bb_obj_list
        self._licence_id = licence_id

    @property
    def image_property_dict(self):
        return {
            "id": self._image_id,
            "file_name": self._image_name,
            "height": self._image_size.height,
            "width": self._image_size.width,
            "license": self._licence_id,
        }

    @property
    def instances_info_dict(self):
        n_instance = len(self._bb_object_list)
        dict_instances = []
        for i in range(n_instance):
            dict_instances.append(
                {
                    "id": self._annotation_id_list[i],
                    "image_id": self._image_id,
                    "segmentation": [self._bb_obj_list[i].bounding_box_polypoints],
                    "iscrowd": 0,
                    "bbox": [self._bb_obj_list[i].bounding_box_coco],
                    "area": self._bb_obj_list[i].area,
                    "category_id": self._class_id_list[i],
                }
            )
        return dict_instances


def cvt_bb_to_instance(bb_obj: BoundingBox):
    x_min, x_max, y_min, y_max = bb_obj.bounding_box_sa
    bbox_instance_sa = {
        "type": "bbox",
        "classId": 1,
        "probability": 100,
        "points": {"x1": x_min, "x2": x_max, "y1": y_min, "y2": y_max},
        "groupId": 0,
        "pointLabels": {},
        "locked": False,
        "visible": True,
        "attributes": [],
    }
    return bbox_instance_sa


def bounding_box_list_to_annotation_dict(image_name: str, bb_list: List[BoundingBox]):
    annotation_info = {
        "instances": [cvt_bb_to_instance(bb_obj) for bb_obj in bb_list],
        "tags": [],
        "metadata": {"version": "1.0.0", "name": image_name, "status": "In progress"},
    }
    return annotation_info


def get_image_pathes(input_dir_pathlib):
    extf = [".jpg", ".png"]
    image_pathes = [path for path in input_dir_pathlib.glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


def load_bounding_boxes_from_txt(ann_txt_path_str, image_size: ImageSize):
    with open(ann_txt_path_str) as f:
        bb_lines = f.readlines()
    bb_list = []
    for i, bb_line in enumerate(bb_lines):
        bb = bb_line.replace("\n", "").split(" ")
        label = int(bb[0])
        if label != Label.Tip:
            continue
        bb_obj = BoundingBox(image_size)
        bb_info = [float(elem) for elem in bb[1:]]
        bb_obj.set_bounding_box_darknet(*bb_info)
        bb_list.append(bb_obj)
    return bb_list


def dump_json(json_path, json_data):
    with open(json_path, "w") as f:
        json.dump(json_data, f)


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

    input_image_size = ImageSize(image_width, image_height) # assume all images have the same image size
    input_image_path_list = get_image_pathes(input_dir_pathlib)
    for input_image_path in tqdm(input_image_path_list):
        # Get image & annotation data path
        image_name = Path(input_image_path).name
        annotation_json_name = image_name + ".json"
        annotation_json_path = str(Path(input_dir_path, annotation_json_name))

        # Load annotation json

        bb_list = load_bounding_boxes_from_txt(annotation_txt_path, input_image_size)
        annotation_dict = bounding_box_list_to_annotation_dict(image_name, bb_list)


        output_image_path = str(Path(output_dir_path, image_name))
        output_annotation_path = str(Path(output_dir_path, image_name + ".json"))
        shutil.copy(input_image_path, output_image_path)
        dump_json(output_annotation_path, annotation_dict)


if __name__ == "__main__":
    main()
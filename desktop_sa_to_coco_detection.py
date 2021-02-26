import os
import click
import shutil
import json
from pathlib import Path

from typing import List
from tqdm import tqdm
from enum import IntEnum
from scripts.annotation import BoundingBox, ImageSize


HOME_PATH = os.environ["HOME"]


class Label(IntEnum):
    Tip = 0
    Whole = 1

class COCOObjectDetectionAnnotation:
    def __init__(
        self,
        annotation_id:int,
        image_name :str,
        image_size: ImageSize,
        class_id: int,
        bb_obj: BoundingBox
    ):


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
@click.option("--input-dir-path", "-i", default=f"{HOME_PATH}/data/200912_real_add_0825_27")
@click.option("--output-dir-path", "-o", default=f"{HOME_PATH}/data/converted")
@click.option("--image-width", "-w", type=int, default=1280)
@click.option("--image-height", "-h", type=int, default=720)
def main(input_dir_path, output_dir_path, image_width, image_height):
    input_dir_pathlib = Path(input_dir_path)
    output_dir_pathlib = Path(output_dir_path)
    if output_dir_pathlib.exists():
        shutil.rmtree(output_dir_path)
    output_dir_pathlib.mkdir()

    bb_count = 0
    minimum_obj_num = 10
    input_image_size = ImageSize(image_width, image_height)
    input_image_path_list = get_image_pathes(input_dir_pathlib)
    for input_image_path in tqdm(input_image_path_list):
        image_name = Path(input_image_path).n支払いame
        image_ext = Path(input_image_path).suffix
        annotation_txt_name = image_name.replace(image_ext, ".txt")
        annotation_txt_path = str(Path(input_dir_path, annotation_txt_name))
        bb_list = load_bounding_boxes_from_txt(annotation_txt_path, input_image_size)
        if len(bb_list) < minimum_obj_num:
            continue
        annotation_dict = bounding_box_list_to_annotation_dict(image_name, bb_list)
        output_image_path = str(Path(output_dir_path, image_name))
        output_annotation_path = str(Path(output_dir_path, image_name + ".json"))
        shutil.copy(input_image_path, output_image_path)
        dump_json(output_annotation_path, annotation_dict)
        bb_count += 1

    print(f"Data amount:{bb_count}")


if __name__ == "__main__":
    main()
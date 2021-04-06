
import json
import click
from pycocotools.coco import COCO
from pathlib import Path
import re
import cv2
import numpy as np
import shutil
from typing import Tuple, List
import pdb

SCRIPT_DIR = str(Path(__file__).parent.resolve())


dict_idx2color = {
    0: (0, 0, 0),
    1: (110, 110, 255),  # Oyagi
    2: (150, 249, 152),  # Aspara
    3: (255, 217, 81),  # Ground
    4: (252, 51, 255),  # Tube
    5: (84, 223, 255),  # Pole
}


def get_image_num(input_data_dir, image_dir_name):
    extf = [".png", ".jpg"]
    image_path_list = [str(path) for path in Path(input_data_dir, image_dir_name).glob("*") if path.suffix in extf]
    return len(image_path_list)


def load_classes_json(classes_json_path):
    with open(classes_json_path, "r") as f:
        classes_json = json.load(f)

    classes_dict = {}
    n_class = len(classes_json)
    for i in range(n_class):
        classes_dict[classes_json[i]["name"]] = classes_json[i]["id"]
    return classes_dict


def get_class_mapper_json(input_data_dir, image_dir_name):
    class_mapper_json_path = str(
        Path(input_data_dir, image_dir_name, "classes_mapper.json")
    )
    with open(class_mapper_json_path, "r") as f:
        class_mapper_json = json.load(f)
    return class_mapper_json


def colorize_mask(mask_image, n_label):
    mask_colorized = np.zeros(
        [mask_image.shape[0], mask_image.shape[1], 3], dtype=np.uint8
    )
    for l in range(n_label+1) :
        mask_indices_lth_label = np.where(mask_image == l)
        mask_colorized[mask_indices_lth_label[0], mask_indices_lth_label[1], :] = dict_idx2color[l]
    return mask_colorized


class COCO_ANNOTATION_MASK_IMAGE:
    def __init__(self, label_name_idx_dict, image_size: Tuple[int, int], prioritized_label_list: List[str] = []):
        self._image_size = image_size  # (width, height)
        self._image_name = None
        self._prioritized_label_list = prioritized_label_list
        self._setting(label_name_idx_dict)

    def _setting(self, label_name_idx_dict):
        self._label_mask_dict = {}
        self._label_to_index = {}
        for l, label_name in enumerate(label_name_idx_dict.keys()):
            self._label_mask_dict[label_name] = None
            self._label_to_index[label_name] = l+1

    def set_mask(self, label_name, label_masks):
        self._label_mask_dict[label_name] = label_masks

    @property
    def image_name(self):
        return self._image_name

    @image_name.setter
    def image_name(self, name):
        self._image_name = name

    @property
    def merged_mask(self):
        image_width = self._image_size[0]
        image_height = self._image_size[1]
        merged_mask_image = np.zeros([image_height, image_width], dtype=np.uint8)

        if len(self._prioritized_label_list) == 0:
            for i, label_name in enumerate(self._label_mask_dict.keys()):
                if self._label_mask_dict[label_name] is not None:
                    masks = self._label_mask_dict[label_name]
                    #mask_image_ith_label = self._label_mask_dict[label_name]
                    for mask_image_ith_label in masks:
                        merged_mask_image[np.where(mask_image_ith_label > 0)] = i + 1
        else:
            for label_name in self._prioritized_label_list:
                if self._label_mask_dict[label_name] is not None:
                    masks = self._label_mask_dict[label_name]
                    for mask_image_ith_label in masks:
                        #mask_image_ith_label = self._label_mask_dict[label_name]
                        merged_mask_image[np.where(mask_image_ith_label > 0)] = self._label_to_index[label_name]
        return merged_mask_image


def get_prioritized_label_list(input_data_dir, class_mapper_json, priority_definition_file, labels):
    with open(priority_definition_file, "r") as f:
        lines_raw = f.readlines()

    lines = [line.replace("\n", "") for line in lines_raw]
    label_list = [label for label in labels]
    prioritized_label_list = lines[::-1]
    return prioritized_label_list


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-mask-dir", "-o", default=f"{SCRIPT_DIR}/mask")
@click.option("--priority-definition-file", "-p", default=f"{SCRIPT_DIR}/cfg/classes_priority.txt")
@click.option("--image-width", "-w", default=1920)
@click.option("--image-height", "-h", default=1080)
@click.option("--classes-json_path", "-c", default=f"{SCRIPT_DIR}/json/classes.json")
@click.option("--json-name", "-j", default="takaune.json")
@click.option("--image-dir-name", "-id", default="image_set")
def main(input_data_dir, output_mask_dir, priority_definition_file, image_width, image_height, classes_json_path, json_name, image_dir_name):
    json_path = str(Path(input_data_dir, json_name))
    annotation_coco = COCO(json_path)
    classes_dict = load_classes_json(classes_json_path)

    image_size = (image_width, image_height)

    n_images = get_image_num(input_data_dir, image_dir_name)
    class_mapper_json = get_class_mapper_json(input_data_dir, image_dir_name)
    classes = class_mapper_json.keys()

    prioritized_label_list = get_prioritized_label_list(
        input_data_dir, class_mapper_json, priority_definition_file, classes
    )

    n_classes = len(classes)
    coco_annotation_mask_obj_list = [
        COCO_ANNOTATION_MASK_IMAGE(class_mapper_json, image_size, prioritized_label_list)
        for i in range(n_images)
    ]

    for class_name in classes:
        #cat_ids = annotation_coco.getCatIds(catNms=[class_name])
        cat_ids = [classes_dict[class_name]]
        img_indices = annotation_coco.getImgIds(catIds=cat_ids)

        for img_ids in img_indices:
            img_ids_tmp = annotation_coco.getImgIds(imgIds=img_ids)
            img_obj = annotation_coco.loadImgs(img_ids_tmp)[0]
            img_name = img_obj["file_name"]
            img_id = img_obj["id"]
            ann_ids = annotation_coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
            anns = annotation_coco.loadAnns(ann_ids)
            coco_annotation_mask_obj_list[img_id].image_name = img_name
            annotation_masks = []
            for ann in anns:
                annotation_masks.append(annotation_coco.annToMask(ann))
            coco_annotation_mask_obj_list[img_id].set_mask(class_name, annotation_masks)

    name_mask_dict = {}
    for coco_annotation_mask_obj in coco_annotation_mask_obj_list:
        merged_mask = coco_annotation_mask_obj.merged_mask.copy()
        name_mask_dict[coco_annotation_mask_obj.image_name] = (merged_mask, colorize_mask(merged_mask, n_classes))

    # Generate mask images
    output_mask_dir_path = Path(output_mask_dir)
    if output_mask_dir_path.exists():
        shutil.rmtree(output_mask_dir_path)
    output_mask_dir_path.mkdir()

    for name_mask in name_mask_dict.keys():
        if name_mask is None:
            continue
        output_colorized_mask_name = "_colorized_" + Path(name_mask).name
        output_mask_name = Path(name_mask).name
        output_colorized_mask_path = str(Path(output_mask_dir_path, output_colorized_mask_name))
        output_mask_path = str(Path(output_mask_dir_path, output_mask_name))

        mask_image, colorized_mask_image = name_mask_dict[name_mask]
        cv2.imwrite(output_mask_path, mask_image)
        cv2.imwrite(output_colorized_mask_path, colorized_mask_image)
        cv2.waitKey(10)


if __name__ == "__main__":
    main()

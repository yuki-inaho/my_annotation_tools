import cv2
import numpy as np
import json
from pathlib import Path
from bidict import bidict
from typing import Tuple, NamedTuple


SCRIPT_DIR = str(Path(__file__).parent)
AnnotationInfo = NamedTuple("AnnotationClass", (("class_name", str), ("class_id", int), ("inference_id", int)))


def hex_to_rgb(color_hex: str) -> Tuple[str]:
    h = color_hex.lstrip("#")
    color_bgr_cv2 = tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))
    color_rgb_cv2 = tuple([color_bgr_cv2[2], color_bgr_cv2[1], color_bgr_cv2[0]])
    return color_rgb_cv2


class AnnotationClassManager:
    def __init__(self, class_json_path: str):
        self._annotation_info_list = []
        self._annotation_bidict = bidict({})
        self._setup(class_json_path)

    def _setup(self, class_json_path: str):
        with open(class_json_path, "r") as f:
            classes_json = json.load(f)

        self._n_class = len(classes_json)
        self._dict_id2name = {}
        self._dict_name2id = {}
        for im1, class_elem in enumerate(classes_json):
            i = im1 + 1  # implicity assumed __background__ label exists
            ann_info = AnnotationInfo(class_name=class_elem["name"], class_id=class_elem["id"], inference_id=i)
            self._annotation_info_list.append(ann_info)
            self._annotation_bidict[class_elem["id"]] = i
            self._dict_id2name[class_elem["id"]] = class_elem["name"]
            self._dict_name2id[class_elem["name"]] = class_elem["id"]

    def class_id_to_inference_id(self, cid):
        return self._annotation_bidict[cid]

    def inference_id_to_class_id(self, iid):
        return self._annotation_bidict.inverse[iid]

    def id2name(self, cid):
        return self._dict_id2name[cid]

    def name2id(self, class_name):
        return self._dict_name2id[class_name]

    @property
    def n_class(self):
        return self._n_class


class LabelColorManager:
    def __init__(self, class_definition_json_path: str):
        class_information_list = self._read_json(class_definition_json_path)
        self._import_class_info(class_information_list)

    def _import_class_info(self, class_information_list):
        self._n_classes = len(class_information_list)
        self._dict_idx2color = {0: (0, 0, 0)}

        # Label index: the number corresponded to neural network output
        # Class ID: the number corresponded to the id written in classes.json
        self._dict_label_index_to_class_id = {0: 0}

        for label_index_m1, class_information in enumerate(class_information_list):
            label_index = label_index_m1 + 1
            self._dict_label_index_to_class_id[label_index] = class_information["id"]
            self._dict_idx2color[label_index] = hex_to_rgb(class_information["color"])

    def _read_json(self, class_definition_json_path):
        with open(class_definition_json_path) as f:
            json_data = json.load(f)
        return json_data

    def label2color(self, id: int):
        return self._dict_idx2color[id]

    def label_index_to_class_id(self, label_index: int):
        return self._dict_label_index_to_class_id[label_index]

    @property
    def n_classes(self):
        return self._n_classes
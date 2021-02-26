import cv2
import numpy as np
import json
from pathlib import Path
from bidict import bidict
from typing import NamedTuple

SCRIPT_DIR = str(Path(__file__).parent)
AnnotationInfo = NamedTuple("AnnotationClass", (("class_name", str), ("class_id", int), ("inference_id", int)))


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

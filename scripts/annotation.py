import numpy as np
from typing import NamedTuple, List


class ImageSize(NamedTuple):
    width: int
    height: int


# TODO: include label info?
class BoundingBox:
    def __init__(self, image_size: ImageSize):
        self._image_size = image_size
        self._category_id = None

    def set_category_id(self, category_id: int):
        self._category_id = category_id

    def set_bounding_box_darknet(self, x, y, w, h):
        self._x_darknet = x
        self._y_darknet = y
        self._w_darknet = w
        self._h_darknet = h
        self._setting_bounding_box_coco_format()

    def _setting_bounding_box_coco_format(self):
        self.x_coco = int(self._x_darknet * self._image_size.width)
        self.y_coco = int(self._y_darknet * self._image_size.height)
        self.w_coco = int(self._w_darknet * self._image_size.width)
        self.h_coco = int(self._h_darknet * self._image_size.height)

    def set_bounding_box_xyxy(self, x_min, y_min, x_max, y_max):
        self._x_darknet = x_min
        self._y_darknet = y_min
        self._w_darknet = x_max - x_min
        self._h_darknet = y_max - y_min
        self._setting_bounding_box_coco_format()

    @property
    def bounding_box_sa(self):
        w_half = int(self.w_coco / 2)
        h_half = int(self.h_coco / 2)
        x_min = self.x_coco - w_half
        y_min = self.y_coco - h_half
        x_max = self.x_coco + w_half
        y_max = self.y_coco + h_half
        return x_min, x_max, y_min, y_max

    @property
    def bounding_box_coco(self):
        return self.x_coco, self.y_coco, self.w_coco, self.h_coco

    @property
    def bounding_box_darknet(self):
        return self.x_darknet, self.y_darknet, self.w_darknet, self.h_darknet

    @property
    def bounding_box_polypoints(self):
        x_min, x_max, y_min, y_max = self.bounding_box_sa

        p1 = np.array([x_min, y_min])
        p2 = np.array([x_min, y_max])
        p3 = np.array([x_max, y_max])
        p4 = np.array([x_max, y_min])

        return np.vstack([p1, p2, p3, p4]).flatten()

    @property
    def area(self):
        x_min, x_max, y_min, y_max = self.bounding_box_sa
        x_length = x_max - x_min
        y_length = y_max - y_min
        return y_length * x_length

    @property
    def category_id(self):
        return self._category_id


class COCOInstanceAnnotation:
    def __init__(
        self,
        image_id: int,
        image_name: str,
        image_size: ImageSize,
        annotation_id_list: List[int],
        bb_object_list: List[BoundingBox],
        licence_id: int = 1,
    ):
        self._image_id = image_id
        self._image_name = image_name
        self._image_size = image_size
        self._annotation_id_list = annotation_id_list
        self._bb_object_list = bb_object_list
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
                    "segmentation": [self._bb_object_list[i].bounding_box_polypoints],
                    "iscrowd": 0,
                    "bbox": [self._bb_object_list[i].bounding_box_coco],
                    "area": self._bb_object_list[i].area,
                    "category_id": self._bb_object_list[i].category_id,
                }
            )
        return dict_instances

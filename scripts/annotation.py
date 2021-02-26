import numpy as np
from typing import NamedTuple


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
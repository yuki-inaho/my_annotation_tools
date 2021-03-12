import os
import cv2
import click
import numpy as np
from pathlib import Path
from scripts.utils import get_image_pathes, mkdir_from_path
from scripts.category import LabelColorManager
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.resolve()


def get_overlay_rgb_image(rgb_image, mask, rgb_rate=0.6, mask_rate=0.4):
    if len(mask.shape) > 2:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        segmentation_overlay_rgb = cv2.addWeighted(rgb_image, rgb_rate, mask, mask_rate, 2.5)
    else:
        mask_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        nonzero_idx = np.where(mask > 0)
        mask_image[nonzero_idx[0], nonzero_idx[1], :] = (0, 0, 255)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        segmentation_overlay_rgb = cv2.addWeighted(rgb_image, rgb_rate, mask_image, mask_rate, 2.5)
    return segmentation_overlay_rgb


def colorize_mask(mask_image: np.ndarray, color_manager: LabelColorManager):
    mask_colorized = np.zeros([mask_image.shape[0], mask_image.shape[1], 3], dtype=np.uint8)
    for l in range(color_manager.n_classes + 1):
        mask_indices_lth_label = np.where(mask_image == l)
        mask_colorized[mask_indices_lth_label[0], mask_indices_lth_label[1], :] = color_manager.label2color(l)
    return mask_colorized


@click.command()
@click.option("--input-data-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-data-dir", "-o", default=f"{SCRIPT_DIR}/output")
@click.option("--annotation-dir-name", "-a", default=f"annotation")
@click.option("--class-definition-json", "-c", default=f"{SCRIPT_DIR}/cfg/classes.json")
def main(input_data_dir, annotation_dir_name, output_data_dir, class_definition_json):
    color_manager = LabelColorManager(class_definition_json)
    input_data_dir_pathlib = Path(input_data_dir)
    input_image_dir_pathlib = input_data_dir_pathlib.joinpath("Image")
    input_annotation_dir_pathlib = input_data_dir_pathlib.joinpath(annotation_dir_name)
    mkdir_from_path(output_data_dir)

    image_path_list = get_image_pathes(input_image_dir_pathlib)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        rgb_image = cv2.imread(image_path)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        annotation_file_name = base_name.replace(".jpg", ".png")
        annotation_file_path = str(input_annotation_dir_pathlib.joinpath(annotation_file_name))
        segmentation_mask = cv2.imread(annotation_file_path, cv2.IMREAD_ANYDEPTH)
        segmentation_mask_colorized = colorize_mask(segmentation_mask, color_manager)
        rgb_image_masked = get_overlay_rgb_image(bgr_image, segmentation_mask_colorized)
        cv2.imwrite(f"{output_data_dir}/{base_name}", rgb_image_masked)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
from colorize_labeled_images import SCRIPT_DIR
import cv2
import click
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scripts.utils import get_image_pathes, mkdir_from_path
from scripts.category import LabelColorManager

SCRIPT_DIR = str(Path().resolve())


@click.command()
@click.option("--input-annotation-dir", "-i", required=True)
@click.option("--fusion-target-annotation-dir", "-t", default=None)
@click.option("--output-annotation-dir", "-o", required=True)
@click.option("--overpaint-region-label", "-lr", type=int, default=2)
@click.option("--overpaint-target-label", "-lt", type=int, default=6)
def main(input_annotation_dir, fusion_target_annotation_dir, output_annotation_dir, overpaint_region_label, overpaint_target_label):
    input_annotation_dir_pathlib = Path(input_annotation_dir)
    output_annotation_dir_pathlib = Path(output_annotation_dir)
    mkdir_from_path(output_annotation_dir_pathlib)

    fusion_mode = fusion_target_annotation_dir is not None
    if fusion_mode:
        fusion_annotation_dir_pathlib = Path(fusion_target_annotation_dir)

    ann_path_list = get_image_pathes(input_annotation_dir_pathlib)
    for annotaiton_mask_path in tqdm(ann_path_list):
        mask_name = Path(annotaiton_mask_path).name
        annotation_mask = cv2.imread(annotaiton_mask_path, cv2.IMREAD_ANYDEPTH)
        annotation_mask_modified = annotation_mask.copy()

        if fusion_mode:
            fusing_target_annotation_mask_path = str(fusion_annotation_dir_pathlib.joinpath(mask_name))
            fusion_target_annotation_mask = cv2.imread(fusing_target_annotation_mask_path, cv2.IMREAD_ANYDEPTH)
            is_overpaint_region = annotation_mask_modified == overpaint_region_label
            is_overpaint_target = fusion_target_annotation_mask == overpaint_target_label
            overpaint_region = is_overpaint_region * is_overpaint_target
        else:
            is_overpaint_region = annotation_mask_modified == overpaint_region_label
            overpaint_region = is_overpaint_region

        annotation_mask_modified[overpaint_region] = overpaint_target_label
        output_annotation_mask_path = str(output_annotation_dir_pathlib.joinpath(mask_name))

        cv2.imwrite(output_annotation_mask_path, annotation_mask_modified)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
import pdb
import os
import sys
import shutil

SCRIPT_DIR = str(Path(__file__).parent)


def generate_output_dir(parent_dir, output_dir_name):
    output_dir_path = Path(parent_dir, output_dir_name)
    output_dir_image_path = Path(output_dir_path, "Image")
    output_dir_annotation_path = Path(output_dir_path, "annotation")
    for opath in [output_dir_path, output_dir_image_path, output_dir_annotation_path]:
        if not opath.exists():
            opath.mkdir()

@click.command()
@click.argument("input-dir-names", nargs=-1)
@click.option("--parent-dir", "-p", default=f"{SCRIPT_DIR}")
@click.option("--output-dir-name", "-o", default="data_merged")
def main(input_dir_names, parent_dir, output_dir_name):
    len_dataset = len(input_dir_names)
    output_dir_path = Path(parent_dir, output_dir_name)
    generate_output_dir(parent_dir, output_dir_name)

    for dname in input_dir_names:
        input_dir_path = Path(parent_dir, dname)
        print(input_dir_path)

        # Copy Images
        input_image_dir = Path(input_dir_path, "Image")
        exts = ['.jpg', '.png']
        image_pathes = sorted([path for path in Path(input_image_dir).glob('*') if path.suffix.lower() in exts])
        image_path_list = [str(image_path) for image_path in image_pathes]

        input_annotation_dir = Path(input_dir_path, "annotation")
        annotation_pathes = sorted([path for path in Path(input_annotation_dir).glob('*.png')])
        annotation_path_list = [str(annotation_path) for annotation_path in annotation_pathes]

        for image_path, annotation_path in zip(image_path_list, annotation_path_list):
            base_image_name = Path(image_path).name
            base_annotation_name = Path(annotation_path).name

            out_image_path = str(Path(output_dir_path, "Image", base_image_name))
            out_annotation_path = str(Path(output_dir_path, "annotation", base_annotation_name))
            shutil.copy(image_path, out_image_path)
            shutil.copy(annotation_path, out_annotation_path)


if __name__ == "__main__":
    main()

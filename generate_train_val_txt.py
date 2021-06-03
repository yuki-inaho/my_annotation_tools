import os
import shutil
import argparse
import numpy as np
from pathlib import Path
from scripts.utils import get_image_pathes

SCRIPT_DIR = str(Path(__file__).parent)
USERNAME = os.getenv("USER")


def get_image_pathes(input_dir_pathlib: Path):
    extf = [".jpg", ".png"]
    image_pathes = [path for path in input_dir_pathlib.glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


def parse_args():
    parser = argparse.ArgumentParser(description="A script to generate train-validation splitted image list files")
    parser.add_argument("--input-project-dir", "-i", default=f"{SCRIPT_DIR}/data")
    parser.add_argument("--output-dir", "-o", default="")
    parser.add_argument("--train-var-rate", "-r", default=0.95)
    parser.add_argument("--default-path", "-p", default=f"/home/{USERNAME}/data")
    return parser


def main(input_project_dir, output_dir, train_var_rate, default_path):
    input_project_dir_pathlib = Path(input_project_dir)
    is_output_same_dir = output_dir == ""
    output_dir_path = output_dir if not is_output_same_dir else input_project_dir
    output_dir_pathlib = Path(output_dir_path)
    if not is_output_same_dir:
        if output_dir_pathlib.exists():
            shutil.rmtree(output_dir_path)
        output_dir_pathlib.mkdir()

    input_image_dir_pathlib = input_project_dir_pathlib.joinpath("Image")

    if not input_image_dir_pathlib.exists():
        input_image_dir_pathlib = input_project_dir_pathlib
    image_path_list = get_image_pathes(input_image_dir_pathlib)
    n_image = len(image_path_list)
    image_indices_shuffle = np.arange(n_image)
    np.random.shuffle(image_indices_shuffle)
    thresh_idx = np.floor(float(n_image) * train_var_rate)

    train_txt_path = str(output_dir_pathlib.joinpath("train.txt"))
    val_txt_path = str(output_dir_pathlib.joinpath("val.txt"))
    if os.path.exists(train_txt_path):
        os.remove(train_txt_path)
    if os.path.exists(val_txt_path):
        os.remove(val_txt_path)

    for i, idx in enumerate(image_indices_shuffle):
        image_path = image_path_list[idx]
        base_name = Path(image_path).name
        out_image_path = str(Path(default_path, "Image", base_name))
        base_mask_name = base_name
        base_mask_name = base_mask_name.replace(".jpg", ".png")
        out_mask_path = str(Path(default_path, "annotation", base_mask_name))
        image_mask_str = f"{out_image_path} {out_mask_path}\n"
        if i < thresh_idx:
            with open(train_txt_path, mode="a") as f:
                f.write(image_mask_str)
        else:
            with open(val_txt_path, mode="a") as f:
                f.write(image_mask_str)


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    main(args.input_project_dir, args.output_dir, args.train_var_rate, args.default_path)

import numpy as np
import click
from pathlib import Path
import os
from scripts.utils import get_image_pathes

SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-project-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-dir", "-o", default=f"{SCRIPT_DIR}")
@click.option("--train-var-rate", "-r", default=0.95)
@click.option("--default-path", "-p", default="/home/yoshi/data")
def main(input_project_dir, output_dir, train_var_rate, default_path):
    input_project_dir_pathlib = Path(input_project_dir)
    output_dir_pathlib = Path(output_dir)
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
    main()
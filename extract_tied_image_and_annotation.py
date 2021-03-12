import json
import shutil
import numpy as np
import click
from pathlib import Path, PosixPath


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def load_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def dump_json(json_path, json_data):
    with open(json_path, "w") as f:
        json.dump(json_data, f, cls=NpEncoder)


def mkdir_from_path(output_image_dir: str):
    output_image_dir_path = Path(output_image_dir)
    if output_image_dir_path.exists():
        shutil.rmtree(output_image_dir)
    output_image_dir_path.mkdir()


def get_image_pathes(input_dir_pathlib: PosixPath):
    extf = [".jpg", ".png"]
    image_pathes = [path for path in input_dir_pathlib.glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list



@click.command()
@click.option("--input-image-dir", "-i", default="Image")
@click.option("--input-annotation-dir", "-a", default="annotation")
@click.option("--output-data-dir", "-o", default="extracted")
def main(input_image_dir, input_annotation_dir, output_data_dir):
    output_data_dir_pathlib = Path(output_data_dir)
    output_data_dir_pathlib.mkdir(exist_ok=True)
    output_data_dir_pathlib.joinpath("Image").mkdir(exist_ok=True)
    output_data_dir_pathlib.joinpath("annotation").mkdir(exist_ok=True)

    input_image_pathlib = Path(input_image_dir)
    input_annotation_pathlib = Path(input_annotation_dir)

    input_ann_path_list = [str(ann_path) for ann_path in input_annotation_pathlib.glob("*.json")]

    for input_ann_path in input_ann_path_list:
        input_ann_name = Path(input_ann_path).name
        if input_ann_name == "classes.json":
            continue
        input_image_name = input_ann_name.replace(".json", "")
        input_image_path = input_image_pathlib.joinpath(input_image_name)
        output_ann_path = output_data_dir_pathlib.joinpath("annotation", input_ann_name)
        output_image_path = output_data_dir_pathlib.joinpath("Image", input_image_name)
        shutil.copy(input_image_path, output_image_path)
        shutil.copy(input_ann_path, output_ann_path)

if __name__ == "__main__":
    main()
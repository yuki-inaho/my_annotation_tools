import json
from pathlib import Path, PosixPath
import shutil


def load_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def dump_json(json_path, json_data):
    with open(json_path, "w") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


def get_image_pathes(input_dir_pathlib: PosixPath):
    extf = [".jpg", ".png"]
    image_pathes = [path for path in input_dir_pathlib.glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


def mkdir_from_path(output_image_dir: str):
    output_image_dir_path = Path(output_image_dir)
    if output_image_dir_path.exists():
        shutil.rmtree(output_image_dir)
    output_image_dir_path.mkdir()

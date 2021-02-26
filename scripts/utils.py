import json


def load_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def dump_json(json_path, json_data):
    with open(json_path, "w") as f:
        json.dump(json_data, f)


def get_image_pathes(input_dir_pathlib):
    extf = [".jpg", ".png"]
    image_pathes = [path for path in input_dir_pathlib.glob("*") if path.suffix in extf]
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list

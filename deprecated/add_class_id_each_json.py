
from os import mkdir
import numpy as np
import copy
from pathlib import Path
import click
from tqdm import tqdm
from scripts.utils import load_json, dump_json, mkdir_from_path
import pdb


SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-json-dir", "-i", default=f"{SCRIPT_DIR}/annotation")
@click.option("--output-json-dir", "-o", default=f"{SCRIPT_DIR}/annotation_out")
def main(input_json_dir, output_json_dir):
    mkdir_from_path(output_json_dir)

    json_pathes = Path(input_json_dir).glob("*.json")
    json_path_list = [str(json_path) for json_path in json_pathes]
    for json_path in tqdm(json_path_list):
        json_name = Path(json_path).name
        json_data = load_json(json_path)
        json_data_modified = copy.deepcopy(json_data)
        n_instances = len(json_data_modified["instances"])
        for i in range(n_instances):
            json_data_modified["instances"][i]["classId"] = 0
        output_json_path = str(Path(output_json_dir, json_name))
        dump_json(output_json_path, json_data_modified)

if __name__ == "__main__":
    main()
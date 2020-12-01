import json
import numpy as np
from pathlib import Path
import click
import pdb

SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-json-path", "-i", default=f"{SCRIPT_DIR}/annotations.json")
@click.option("--output-json-path", "-o", default=f"{SCRIPT_DIR}/annotations_out.json")
def main(input_json_path, output_json_path):
    with open(input_json_path, 'r') as f:
        annotation_data = json.load(f)

    for key in annotation_data.keys():
        elem_list_wo_bbox = []
        elem_list = annotation_data[key]
        for elem in elem_list:
            if elem["type"] in ["polygon", "meta"] :
                elem_list_wo_bbox.append(elem)
        annotation_data[key] = elem_list_wo_bbox

    with open(output_json_path, 'w') as f:
        json.dump(annotation_data, f)

if __name__ == "__main__":
    main()
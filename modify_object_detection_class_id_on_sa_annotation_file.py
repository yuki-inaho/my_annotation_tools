import os
import click
import copy
from scripts.utils import load_json, dump_json


HOME_PATH = os.environ["HOME"]


@click.command()
@click.option("--input-json-path", "-i", default=f"{HOME_PATH}/superannotate_preupload/annotation.json")
@click.option("--output-json-path", "-o", default=f"{HOME_PATH}/superannotate_preupload/annotation_modified.json")
def main(input_json_path, output_json_path):
    json_data = load_json(input_json_path)
    modified_json_data = copy.deepcopy(json_data)

    dict_id2label = {}
    category_name_list = []
    for category in json_data["categories"]:
        dict_id2label[category["id"]] = category["name"]
        category_name_list.append(category["name"])

    dict_label_to_modified_index = {}
    if "pseudo" in category_name_list:
        for new_index_m1, category_name in enumerate(category_name_list):
            if category_name == "pseudo":
                dict_label_to_modified_index[category_name] = 0
                modified_json_data["categories"][new_index_m1]["id"] = 0
            else:
                dict_label_to_modified_index[category_name] = new_index_m1 + 1
                modified_json_data["categories"][new_index_m1]["id"] = new_index_m1 + 1
    else:
        for new_index, category_name in enumerate(category_name_list):
            dict_label_to_modified_index[category_name] = new_index
            modified_json_data["categories"][new_index]["id"] = new_index

    n_instances = len(json_data["annotations"])
    for i in range(n_instances):
        category_id = json_data["annotations"][i]["category_id"]
        modified_json_data["annotations"][i]["category_id"] = dict_label_to_modified_index[dict_id2label[category_id]]

    dump_json(output_json_path, modified_json_data)


if __name__ == "__main__":
    main()
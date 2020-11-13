import os
import shutil
import json
import click
import zipfile
import pdb
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any, NamedTuple

AnnotatedDataTuple = NamedTuple('AnnotatedDataTuple', [
     ('image_path', str),
     ('annotation_dict', Dict[str, Any]),
])

ConfigJsonTuple = NamedTuple('ConfigJsonTuple', [
     ('classes_json', Dict[str, Any]),
     ('config_json', Dict[str, Any]),
])


SCRIPT_DIR = Path(__file__).parent
HOME = os.environ["HOME"]                                                                                                                            

def generate_working_directory(tempolary_working_directory):
    if Path(tempolary_working_directory).exists():
        shutil.rmtree(tempolary_working_directory)
    Path(tempolary_working_directory).mkdir()


def get_input_zip_filepath_list(input_zip_directory):
    zip_file_pathes = Path(input_zip_directory).glob("*.zip")
    zip_filepath_list = [str(zip_file_path) for zip_file_path in zip_file_pathes]
    return zip_filepath_list


def extract_zip_files(zip_filepath_list, temporal_working_directory):
    output_dir_path_list = []
    for zip_file_path in tqdm(zip_filepath_list):
        base_name = Path(zip_file_path).name
        base_name = base_name.replace(".zip", "")
        output_dir_path = Path(temporal_working_directory, base_name)
        output_dir_path.mkdir()
        with zipfile.ZipFile(zip_file_path) as existing_zip:
            existing_zip.extractall(output_dir_path)        
        output_dir_path_list.append(output_dir_path_list)
    return output_dir_path_list


def read_json(json_path):
    with open(json_path, "r") as f:
        json_data = json.load(f)
    return json_data


def write_json(json_path, json_data):
    with open(json_path, "w") as f:
        json.dump(json_data, f)


def read_annotation_json_from_directory(directory_path):
    annotation_json_path = str(Path(directory_path, "annotations.json"))
    annotation_json = read_json(annotation_json_path)
    return annotation_json


def remove_non_annotated_image_name(annotation_image_name_list_raw, annotation_json):
    annotation_image_name_list = []
    for annotation_image_name in annotation_image_name_list_raw:
        if len(annotation_json[annotation_image_name]) > 1:
            annotation_image_name_list.append(annotation_image_name)
    return annotation_image_name_list


def generate_annotated_data_tuple(annotation_image_name_list, annotation_json, output_dir_path):
    annotated_data_tuple_list = []
    for annotation_image_name in annotation_image_name_list:
        ann_data_tuple = AnnotatedDataTuple(
            image_path = str(Path(output_dir_path, "images", annotation_image_name)),
            annotation_dict = annotation_json[annotation_image_name]
        )
        annotated_data_tuple_list.append(ann_data_tuple)
    return annotated_data_tuple_list


def add_annotated_data_tuple_to_dict_without_duplication(annotated_data_tuple_list, dict_image_name_to_annotated_data_tuple):
    for annotated_data_tuple in annotated_data_tuple_list:
        image_path = str(Path(annotated_data_tuple.image_path).name)
        if not image_path in dict_image_name_to_annotated_data_tuple.keys():
            dict_image_name_to_annotated_data_tuple[image_path] = annotated_data_tuple


def get_classes_and_config_json_tuple(directory_path):
    classes_json_path = str(Path(directory_path, "classes.json"))
    classes_json = read_json(classes_json_path)

    config_json_path = str(Path(directory_path, "config.json"))
    config_json = read_json(config_json_path)

    config_json_tuple = ConfigJsonTuple(
        classes_json = classes_json,
        config_json = config_json
    )
    return config_json_tuple


def copy_images(dict_image_name_to_annotated_data_tuple, output_image_dir):
    for image_name in dict_image_name_to_annotated_data_tuple.keys():
        annotated_data_tuple = dict_image_name_to_annotated_data_tuple[image_name]
        image_path = annotated_data_tuple.image_path
        output_image_path = str(Path(output_image_dir, image_name))
        shutil.copy(image_path, output_image_path)


def merge_annotated_data(dict_image_name_to_annotated_data_tuple):    
    merged_annotation_dict = {}
    for image_name in dict_image_name_to_annotated_data_tuple.keys():
        annotated_data_tuple = dict_image_name_to_annotated_data_tuple[image_name]
        merged_annotation_dict[image_name] = annotated_data_tuple.annotation_dict
    return merged_annotation_dict


def generate_merged_annotation(config_json_tuple, dict_image_name_to_annotated_data_tuple, output_annotation_directory):
    # Mkdir
    output_annotation_directory_path = Path(output_annotation_directory)
    if output_annotation_directory_path.exists():
        shutil.rmtree(output_annotation_directory)
    output_annotation_directory_path.mkdir()
    output_image_dir = Path(output_annotation_directory_path, "images")
    output_image_dir.mkdir()

    # Generate config jsons
    classes_json_path = str(Path(output_annotation_directory_path, "classes.json"))
    config_json_path = str(Path(output_annotation_directory_path, "config.json"))
    write_json(classes_json_path, config_json_tuple.classes_json)
    write_json(config_json_path, config_json_tuple.config_json)

    # Copy Images
    copy_images(dict_image_name_to_annotated_data_tuple, output_image_dir)

    # Merge annotation dictionary data
    merged_annotation_dict = merge_annotated_data(dict_image_name_to_annotated_data_tuple)
    annotation_json_path = str(Path(output_annotation_directory_path, "annotations.json"))
    write_json(annotation_json_path, merged_annotation_dict)


@click.command()
@click.option("--input-zip-directory", "-i", default=f"{HOME}/data/sa_zips_to_concat")
@click.option("--temporal-working-directory", "-tmp", default=f"{SCRIPT_DIR}/concate_work_space")
@click.option("--output-annotation-directory", "-i", default=f"{HOME}/data/sa_zips_to_concat")
def main(input_zip_directory, temporal_working_directory, output_annotation_directory):

    #generate_working_directory(temporal_working_directory)
    #zip_filepath_list = get_input_zip_filepath_list(input_zip_directory)
    #output_dir_path_list = extract_zip_files(zip_filepath_list, temporal_working_directory)
    output_dir_pathes = Path(temporal_working_directory).glob("*")
    output_dir_path_list = [str(output_dir_path) for output_dir_path in output_dir_pathes]
    output_dir_path_list.sort()

    dict_image_name_to_annotated_data_tuple = {}
    for i, output_dir_path in enumerate(output_dir_path_list):
        if i==0:
            config_json_tuple = get_classes_and_config_json_tuple(output_dir_path)

        annotation_json = read_annotation_json_from_directory(output_dir_path)
        annotation_image_name_list = [annotation_img_name for annotation_img_name in annotation_json.keys()]
        annotation_image_name_list = remove_non_annotated_image_name(annotation_image_name_list, annotation_json)
        annotated_data_tuple_list = generate_annotated_data_tuple(annotation_image_name_list, annotation_json, output_dir_path)

        add_annotated_data_tuple_to_dict_without_duplication(
            annotated_data_tuple_list, dict_image_name_to_annotated_data_tuple
        )

    generate_merged_annotation(config_json_tuple, dict_image_name_to_annotated_data_tuple, output_annotation_directory)

if __name__ == "__main__":
    main()
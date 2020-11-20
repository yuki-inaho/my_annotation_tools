import click
from pathlib import Path
from tqdm import tqdm
import shutil


SCRIPT_DIR = str(Path(__file__).parent)


def get_image_path_list(input_project_dir):
    exts = ['.jpg', '.png']
    image_pathes = sorted([path for path in Path(input_project_dir, "images").glob('*') if path.suffix.lower() in exts])
    image_path_list = [str(image_path) for image_path in image_pathes]
    return image_path_list


@click.command()
@click.option("--input-project-dir", "-i", default=f"{SCRIPT_DIR}/data")
@click.option("--output-project-dir", "-o", default=f"{SCRIPT_DIR}/data_web")
def main(input_project_dir, output_project_dir):
    image_path_list = get_image_path_list(input_project_dir)
    for image_path in tqdm(image_path_list):
        base_name = Path(image_path).name
        write_name = base_name + "_____save.png"
        shutil.copy(image_path, str(Path(output_project_dir, write_name)))
        cv2.waitKey(10)

if __name__ == "__main__":
    main()

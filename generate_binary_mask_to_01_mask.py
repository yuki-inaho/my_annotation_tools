import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
import pdb


SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-mask-dir", "-i", default=f"{SCRIPT_DIR}/data/mask_raw")
@click.option("--output-mask-dir", "-o", default=f"{SCRIPT_DIR}/data/mask")
def main(input_mask_dir, output_mask_dir):
    mask_pathes = Path(input_mask_dir).glob("*.png")
    mask_path_list = [str(mask_path) for mask_path in mask_pathes]

    for mask_path in tqdm(mask_path_list):
        mask = cv2.imread(mask_path, cv2.IMREAD_ANYDEPTH)
        mask = (mask / 255).astype(np.uint8)
        base_name = Path(mask_path).name
        cv2.imwrite(str(Path(output_mask_dir, base_name)), mask)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()

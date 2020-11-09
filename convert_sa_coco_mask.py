import cv2
import numpy as np
import click
from pathlib import Path
from tqdm import tqdm
import pdb


SCRIPT_DIR = str(Path(__file__).parent)

@click.command()
@click.option("--input-image-dir", "-c", default=f"{SCRIPT_DIR}/data/mask_raw")
@click.option("--output-image-dir", "-c", default=f"{SCRIPT_DIR}/data/mask")
def main(input_image_dir, output_image_dir):
    sa_mask_pathes = Path(input_image_dir).glob("*.png")
    sa_mask_path_list = [str(mask_path) for mask_path in sa_mask_pathes]

    for sa_mask_path in tqdm(sa_mask_path_list):
        sa_mask = cv2.imread(sa_mask_path)
        mask = np.zeros([sa_mask.shape[0], sa_mask.shape[1]], dtype=np.uint8)
        for i in range(3):
            mask[np.where(sa_mask[:,:,i]>0)] = 255
        mask = cv2.resize(mask, (1920,1080), interpolation=cv2.INTER_NEAREST)

        #mask = cv2.cvtColor(sa_mask, cv2.COLOR_RGB2GRAY)
        #mask *= 255
        base_name = Path(sa_mask_path).name
        modified_name = base_name[:-4]
        cv2.imwrite(str(Path(output_image_dir, modified_name)), mask)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()


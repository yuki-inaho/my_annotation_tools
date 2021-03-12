import cv2
import click
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scripts.ssim import SSIM
from scripts.utils import mkdir_from_path, get_image_pathes
import matplotlib.pyplot as plt


def draw_ssim_hist(ssim_list):
    n, bins = np.histogram(ssim_list, bins=100, range=(0, 1.0), weights=None, normed=False)
    fig = plt.figure()
    plt.step(bins[:-1], n, where='post')
    fig.savefig("ssim.png")


@click.command()
@click.option("--input-image-dir", "-i", required=True)
@click.option("--output-image-dir", "-o", required=True)
def main(input_image_dir, output_image_dir):
    input_image_dir_pathlib = Path(input_image_dir)
    output_image_dir_pathlib = Path(output_image_dir)
    mkdir_from_path(output_image_dir)

    input_image_pathes = get_image_pathes(input_image_dir_pathlib)
    n_input_images = len(input_image_pathes)
    ssim_list = []
    for i in tqdm(range(1, n_input_images)):
        prev_image = cv2.imread(input_image_pathes[i - 1])
        curr_image = cv2.imread(input_image_pathes[i])
        ssim_value = SSIM(prev_image, curr_image, max_value=None, win_size=7)
        if ssim_value < 0.28:
            input_image_pathlib = Path(input_image_pathes[i])
            input_image_name = input_image_pathlib.name
            output_image_pathlib = output_image_dir_pathlib.joinpath(input_image_name)
            shutil.copy(str(input_image_pathlib), str(output_image_pathlib))
        #ssim_list.append(ssim_value)


if __name__ == "__main__":
    main()
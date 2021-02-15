import os
import click
from pathlib import Path
from tqdm import tqdm
import pdb
import re

SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-txt", "-i", default=f"{SCRIPT_DIR}/data/train.txt")
@click.option("--output-txt", "-o", default=f"{SCRIPT_DIR}/data/train_reviced.txt")
def main(input_txt, output_txt):
    if os.path.exists(output_txt):
        os.remove(output_txt)

    with open(input_txt, mode="r") as f_in:
        lines = f_in.readlines()

    for line in tqdm(lines):
        img_info, ann_info = line.split(" ")
        img_info = img_info.replace(".png", ".jpg")
        new_line = f"{img_info} {ann_info}"
        with open(output_txt, mode="a") as f_out:
            f_out.write(new_line)

if __name__ == "__main__":
    main()

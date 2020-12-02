
import superannotate as sa
from pathlib import Path
import click

@click.command()
@click.option("--input-dir", "-i", required=True)
@click.option("--output-dir", "-o", default="")
@click.option("--dataset-name", "-n", default="takaune")
def main(input_dir, output_dir, dataset_name):
    if output_dir == "":
        base_name = Path(input_dir).name
        dir_name = str(Path(input_dir).parent)
        output_dir = str(Path(dir_name, f"{base_name}_coco"))

    sa.export_annotation_format(
        input_dir=input_dir,
        output_dir=output_dir,
        dataset_format="COCO",
        dataset_name=dataset_name,
        project_type="Vector",
        task="instance_segmentation",
        platform="Desktop"
    )


if __name__ == "__main__":
    main()
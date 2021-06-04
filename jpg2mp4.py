import click
import ffmpeg
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).parent)


@click.command()
@click.option("--input-images-dir", "-i", default=f"{SCRIPT_DIR}/output")
@click.option("--output-mp4-path", "-o", default=f"{SCRIPT_DIR}/output.mp4")
@click.option("--fps", type=float, default=2)
@click.option("--target-file-size-mib", type=int, default=9)  # 9MiB
def main(input_images_dir, output_mp4_path, fps, target_file_size_mib):
    n_frames = len([str(elem) for elem in Path(input_images_dir).glob("*.jpg")])
    duration = float(n_frames) * fps
    bitrate = "{}k".format(int(target_file_size_mib * 8192 / (0.70 * duration)))

    out, err = (
        ffmpeg.input(str(Path(input_images_dir).joinpath("*.jpg")), pattern_type="glob", framerate=fps)
        .output(output_mp4_path, **{"c:v": "libx264", "b:v": bitrate, "b:a": 0})
        .run(capture_stdout=True)
    )


if __name__ == "__main__":
    main()

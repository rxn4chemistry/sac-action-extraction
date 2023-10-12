from pathlib import Path
from typing import Iterable

import click


def concatenate_files(input_paths: Iterable[Path], output_path: Path) -> None:
    with open(str(output_path), "wt") as f_out:
        for input_path in input_paths:
            with open(str(input_path), "rt") as f_in:
                for line in f_in:
                    f_out.write(line)


@click.command()
@click.option("--dir1", required=True, help="Directory with annotations 1", type=Path)
@click.option("--dir2", required=True, help="Directory with annotations 2", type=Path)
@click.option(
    "--combined",
    required=True,
    help="Directory where to save combined annotations",
    type=Path,
)
def concatenate_annotation_splits(dir1: Path, dir2: Path, combined: Path):
    """Concatenate the annotation splits from different rounds (currently limited to two)."""
    combined.mkdir(exist_ok=True)

    filenames = [
        f"{kind}-{split}.txt"
        for kind in ["src", "tgt"]
        for split in ["train", "valid", "test"]
    ]

    for filename in filenames:
        concatenate_files(
            input_paths=[dir1 / filename, dir2 / filename],
            output_path=combined / filename,
        )


if __name__ == "__main__":
    concatenate_annotation_splits()

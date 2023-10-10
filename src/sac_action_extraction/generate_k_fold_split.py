import itertools
import logging
from pathlib import Path

import click
import numpy as np
from rxn.utilities.files import (
    dump_list_to_file,
    ensure_directory_exists_and_is_empty,
    iterate_lines_from_file,
)
from rxn.utilities.logging import setup_console_logger
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    "--input_dir",
    "-i",
    required=True,
    help="Directory containing the data to augment (src-train.txt, src-valid.txt, etc.)",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Directory where to save the splits (will create several directories there)",
)
@click.option(
    "--n_splits",
    "-n",
    type=int,
    default=5,
    help="Number of splits for k-fold data generation",
)
def main(input_dir: Path, output_dir: Path, n_splits: int) -> None:
    """Generate k-fold splits from an initially split dataset.
    Note that the original train-valid split is fully ignored."""
    setup_console_logger()
    ensure_directory_exists_and_is_empty(output_dir)

    logger.info(f'Loading data from "{input_dir}"...')
    input_array = np.array(
        list(
            itertools.chain(
                iterate_lines_from_file(input_dir / "src-train.txt"),
                iterate_lines_from_file(input_dir / "src-valid.txt"),
            )
        )
    )
    output_array = np.array(
        list(
            itertools.chain(
                iterate_lines_from_file(input_dir / "tgt-train.txt"),
                iterate_lines_from_file(input_dir / "tgt-valid.txt"),
            )
        )
    )
    logger.info(
        f'Loading data from "{input_dir}"... Done - {len(input_array)} items loaded'
    )

    kf = KFold(n_splits=n_splits)

    for i, (train_index, valid_index) in enumerate(
        kf.split(input_array, output_array), 1
    ):
        cv_dir = output_dir / f"cv_{i}"
        cv_dir.mkdir()

        logger.info(
            f'Saving fold {i} to "{cv_dir}" - with {len(train_index)} '
            f"train and {len(valid_index)} validation samples."
        )
        dump_list_to_file(input_array[train_index], cv_dir / "src-train.txt")
        dump_list_to_file(input_array[valid_index], cv_dir / "src-valid.txt")
        dump_list_to_file(output_array[train_index], cv_dir / "tgt-train.txt")
        dump_list_to_file(output_array[valid_index], cv_dir / "tgt-valid.txt")


if __name__ == "__main__":
    main()

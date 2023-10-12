import random
from pathlib import Path

import click
from paragraph2actions.data_augmentation import augment_annotations
from paragraph2actions.default_converters import default_action_converters
from rxn.utilities.logging import setup_console_logger

from .sac_converters import default_sac_converters


@click.command()
@click.option(
    "--data_dir",
    "-d",
    required=True,
    help="Directory with files to augment",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--value_lists_dir",
    "-v",
    required=True,
    help="Directory with durations, temperatures, etc.",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output_dir",
    "-o",
    required=True,
    help="Output directory",
    type=click.Path(path_type=Path),
)
@click.option(
    "--n_augmentations",
    "-n",
    default=10,
    help="Number of augmentations per sample",
    type=int,
)
def main(data_dir: Path, value_lists_dir: Path, output_dir: Path, n_augmentations: int):
    """Augment the training samples."""
    setup_console_logger()
    random.seed(42)

    single_action_converters = default_sac_converters() + default_action_converters()

    augment_annotations(
        data_dir=data_dir,
        value_lists_dir=value_lists_dir,
        output_dir=output_dir,
        single_action_converters=single_action_converters,
        n_augmentations=n_augmentations,
    )


if __name__ == "__main__":
    main()

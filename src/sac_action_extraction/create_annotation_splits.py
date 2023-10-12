import random
from pathlib import Path

import click
from paragraph2actions.data_splitting import create_annotation_splits
from paragraph2actions.default_converters import default_action_converters
from rxn.utilities.logging import setup_console_logger

from .sac_converters import default_sac_converters


@click.command()
@click.option(
    "--annotated_data_dir",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Directory with original "sentences.txt" and "actions.txt", or "annotated.json"',
)
@click.option(
    "--output_dir",
    "-o",
    required=True,
    help="Output directory",
    type=click.Path(path_type=Path),
)
@click.option(
    "--valid_fraction", "-v", default=0.2, type=float, help="% for validation split"
)
@click.option("--test_fraction", "-t", default=0.2, type=float, help="% for test split")
def main(
    annotated_data_dir: Path,
    output_dir: Path,
    valid_fraction: float,
    test_fraction: float,
) -> None:
    """From the files with annotations, create the annotation splits (train, valid,
    test) without augmentation yet."""
    setup_console_logger()

    random.seed(42)

    single_action_converters = default_sac_converters() + default_action_converters()

    create_annotation_splits(
        annotated_data_dir=annotated_data_dir,
        output_dir=output_dir,
        single_action_converters=single_action_converters,
        valid_fraction=valid_fraction,
        test_fraction=test_fraction,
    )


if __name__ == "__main__":
    main()

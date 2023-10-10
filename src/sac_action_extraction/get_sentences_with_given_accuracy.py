from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List, Tuple

import click
import textdistance
from rxn.utilities.files import dump_list_to_file, load_list_from_file


@click.command()
@click.option(
    "--sentences_file",
    "-s",
    required=True,
    help="File containing the sentences",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--ground_truth_file",
    "-g",
    required=True,
    help="File containing the ground truth",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--prediction_file",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="File containing the translations to compare with the ground truth",
)
@click.option(
    "--output_dir",
    "-o",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory where to save the files",
)
@click.option(
    "--cuts",
    "-c",
    type=int,
    multiple=True,
    help="Separators to measure accuracy at (for instance: 100, 50, 0)",
)
def main(
    sentences_file: Path,
    ground_truth_file: Path,
    prediction_file: Path,
    output_dir: Path,
    cuts: Tuple[int, ...],
) -> None:
    """From a file of ground truth actions and predicted actions, split them
    into depending on the levenshtein match, and save into a directory accordingly."""

    sentences = load_list_from_file(sentences_file)
    ground_truth = load_list_from_file(ground_truth_file)
    predictions = load_list_from_file(prediction_file)

    cuts_sorted = sorted(cuts, reverse=True)

    sentences_split_by_cut: DefaultDict[
        int, List[Tuple[str, str, str, float]]
    ] = defaultdict(list)

    for sentence, gt, pred in zip(sentences, ground_truth, predictions):
        similarity = textdistance.levenshtein.normalized_similarity(gt, pred)

        for cut in cuts_sorted:
            if 100 * similarity >= cut:
                values = sentence, gt, pred, similarity
                sentences_split_by_cut[cut].append(values)
                break
        else:
            print(f"Warning: Similarity of {similarity:.2f} not assigned")

    output_dir.mkdir(exist_ok=True, parents=True)
    for cut, values_lists in sentences_split_by_cut.items():
        dump_list_to_file(
            [x[0] for x in values_lists], output_dir / f"sentences_{cut}.txt"
        )
        dump_list_to_file(
            [x[1] for x in values_lists], output_dir / f"ground_truth_{cut}.txt"
        )
        dump_list_to_file(
            [x[2] for x in values_lists], output_dir / f"predictions_{cut}.txt"
        )
        dump_list_to_file(
            [str(x[3]) for x in values_lists], output_dir / f"similarities_{cut}.txt"
        )


if __name__ == "__main__":
    main()

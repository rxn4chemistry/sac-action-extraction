import logging
from pathlib import Path
from typing import List, Tuple

import click
from action_sequences.paragraph_to_actions.analysis import (
    levenshtein_similarity,
    partial_accuracy,
)
from action_sequences.utils.onmt.translator_with_sentencepiece import (
    TranslatorWithSentencePiece,
)
from rxn.utilities.files import load_list_from_file
from rxn.utilities.logging import setup_console_logger

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    "--sp_model",
    "-s",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SentencePiece model",
)
@click.option(
    "--valid_src",
    "-vs",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Validation src",
)
@click.option(
    "--valid_tgt",
    "-vt",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Validation tgt",
)
@click.option(
    "--results_output_txt",
    "-r",
    type=click.Path(writable=True, path_type=Path),
    help="Where to save the results",
)
@click.argument(
    "model_dirs",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    nargs=-1,
)
def main(
    sp_model: Path,
    valid_src: Path,
    valid_tgt: Path,
    results_output_txt: Path,
    model_dirs: Tuple[Path, ...],
) -> None:
    """Calculate metrics for predictions generated by one or several translation models"""
    setup_console_logger()

    src = load_list_from_file(valid_src)
    tgt = load_list_from_file(valid_tgt)

    accuracies: List[Tuple[Path, float, float, float]] = []
    for model_dir in model_dirs:
        for model in model_dir.glob("*.pt"):
            logger.info(f"Evaluating model: {model}")
            translator = TranslatorWithSentencePiece(
                str(model), sentencepiece_model=str(sp_model)
            )
            preds = translator.translate_sentences(src)

            similarity = levenshtein_similarity(tgt, preds)
            acc100 = partial_accuracy(tgt, preds, 1.0)
            acc90 = partial_accuracy(tgt, preds, 0.9)
            accuracies.append((model, similarity, acc100, acc90))

    sorted_results = sorted(accuracies, key=lambda x: x[1], reverse=True)
    with open(results_output_txt, "w") as f:
        for model, similarity, acc100, acc90 in sorted_results:
            f.write(
                f"{model}: lev {similarity:.4f}, acc100 {acc100:.4f}, acc90 {acc90:.4f}\n"
            )

    logger.info(f'Wrote results to "{results_output_txt}"')


if __name__ == "__main__":
    main()

import json
from pathlib import Path

import click
import pandas as pd
from paragraph2actions.analysis import (
    action_string_validity,
    full_sentence_accuracy,
    levenshtein_similarity,
    modified_bleu,
    partial_accuracy,
)
from paragraph2actions.default_converters import default_action_converters
from paragraph2actions.readable_converter import ReadableConverter
from paragraph2actions.translator import Translator
from rxn.utilities.files import load_list_from_file

from .sac_converters import default_sac_converters


def load_converter() -> ReadableConverter:
    single_action_converters = default_sac_converters() + default_action_converters()
    return ReadableConverter(single_action_converters=single_action_converters)


@click.command()
@click.option(
    "--model_json",
    "-mj",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Specifying which models to consider",
)
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
    "--output_csv",
    "-o",
    type=click.Path(writable=True, path_type=Path),
    help="Where to save the resutls (as CSV)",
)
def main(
    model_json: Path,
    sp_model: Path,
    valid_src: Path,
    valid_tgt: Path,
    output_csv: Path,
) -> None:
    """Calculate metrics for predictions generated by one or several translation models"""

    src = load_list_from_file(valid_src)
    tgt = load_list_from_file(valid_tgt)
    converter = load_converter()

    with open(model_json, "r") as f:
        models = json.load(f)

    metrics = []
    for model, checkpoint in models.items():
        # note: example: "combined_data_646", "11000"
        print(f"Evaluating model {model}, checkpoint {checkpoint}")
        exp_dir = Path(f"exp_{model}")
        checkpoint_path = exp_dir / "models" / f"model_lr_0.20_step_{checkpoint}.pt"
        translator = Translator(str(checkpoint_path), sentencepiece_model=str(sp_model))
        preds = translator.translate_sentences(src)

        metrics.append(
            {
                "model": str(model),
                "Full sentence accuracy": full_sentence_accuracy(tgt, preds),
                "String validity": action_string_validity(preds, converter=converter),
                "BLEU": modified_bleu(tgt, preds),
                "Levenshtein": levenshtein_similarity(tgt, preds),
                "100% accuracy": partial_accuracy(tgt, preds, 1.0),
                "90% accuracy": partial_accuracy(tgt, preds, 0.9),
                "75% accuracy": partial_accuracy(tgt, preds, 0.75),
            }
        )
    df = pd.DataFrame(data=metrics)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
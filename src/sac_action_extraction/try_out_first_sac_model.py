# paragraph to actions on aCe@ETH data

import sys
from pathlib import Path
from typing import TextIO

import click
from paragraph2actions.paragraph_translator import ParagraphTranslator

from .utils import load_converter


@click.command()
@click.argument("input_file", type=click.File(mode="r"), default=sys.stdin)
@click.argument("output_file", type=click.File(mode="w"), default=sys.stdout)
def main(input_file: TextIO, output_file: TextIO) -> None:
    """try paragraph translator"""
    init_dir = Path("/Users/ava/Documents/client_projects/ace_eth/20220902_retraining")
    model = init_dir / "model_lr_0.20_step_9000.pt"
    sp_model = init_dir / "original_models" / "sp_model.model"

    converter = load_converter()

    paragraph_translator = ParagraphTranslator(
        translation_model=str(model),
        sentencepiece_model=str(sp_model),
        action_string_converter=converter,
    )

    for line in input_file:
        line = line.strip()

        # Canonicalize the SMILES, handle exception if needed
        try:
            result = paragraph_translator.extract_paragraph(line)
            output_file.write(f"{result.actions}\n")
            output_file.write(f"{converter.actions_to_string(result.actions)}\n")
        except Exception as e:
            print("Error")
            print(e)


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
from typing import TextIO, Tuple

import click
from paragraph2actions.default_converters import default_action_converters
from paragraph2actions.paragraph_translator import ParagraphTranslator
from paragraph2actions.readable_converter import ReadableConverter

from .sac_converters import default_sac_converters


@click.command()
@click.option(
    "--models",
    "-m",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Translation model file(s). If multiple are given, will be an ensemble model.",
)
@click.option(
    "--sentencepiece_model",
    "-s",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SentencePiece model file (typically ``sp_model.model``)",
)
@click.argument("input_file", type=click.File(mode="r"), default=sys.stdin)
@click.argument("output_file", type=click.File(mode="w"), default=sys.stdout)
def main(
    models: Tuple[Path, ...],
    sentencepiece_model: Path,
    input_file: TextIO,
    output_file: TextIO,
) -> None:
    """Interactive extraction of SAC synthesis actions.

    Input and output can be either stdin/stdout, or input file and output file.
    """

    single_action_converters = default_sac_converters() + default_action_converters()
    converter = ReadableConverter(single_action_converters=single_action_converters)

    paragraph_translator = ParagraphTranslator(
        translation_model=[str(m) for m in models],
        sentencepiece_model=str(sentencepiece_model),
        action_string_converter=converter,
    )

    for line in input_file:
        line = line.strip()

        # Canonicalize the SMILES, handle exception if needed
        try:
            result = paragraph_translator.extract_paragraph(line)
            output_file.write("Extracted actions as Python objects:\n")
            output_file.write(f"{result.actions}\n")
            output_file.write("Extracted actions in readable format:\n")
            output_file.write(f"{converter.actions_to_string(result.actions)}\n\n")
        except Exception as e:
            output_file.write(f"\nError: {e}\n")


if __name__ == "__main__":
    main()

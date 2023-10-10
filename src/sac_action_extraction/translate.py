# LICENSED INTERNAL CODE. PROPERTY OF IBM.
# IBM Research Zurich Licensed Internal Code
# (C) Copyright IBM Corp. 2021
# ALL RIGHTS RESERVED

import logging
from typing import Tuple

import click
import sentencepiece  # noqa: must be imported before torch to avoid problems on the cluster.
from action_sequences.utils.misc import setup_basic_logger
from action_sequences.utils.onmt.translator_with_sentencepiece import (
    TranslatorWithSentencePiece,
)
from rxn.onmt_utils.torch_utils import set_num_threads
from rxn.utilities.files import dump_list_to_file, load_list_from_file

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@click.command()
@click.option(
    "--translation_models",
    "-t",
    multiple=True,
    help="Translation model file. If multiple are given, will " "be an ensemble model.",
)
@click.option(
    "--sentencepiece_model", "-p", required=True, help="SentencePiece model file"
)
@click.option("--src_file", "-s", required=True, help="File to translate")
@click.option("--output_file", "-o", required=True, help="Where to save translation")
@click.option("--num_threads", "-n", default=4, help="Number of CPU threads to use")
def translate_actions(
    translation_models: Tuple[str, ...],
    sentencepiece_model: str,
    src_file: str,
    output_file: str,
    num_threads: int,
):
    """
    Translate a text with an OpenNMT model.

    This script, in addition to the OpenNMT translation, adds pre-processing and
    post-processing in the form of tokenization and de-tokenization with sentencepiece.
    """
    setup_basic_logger()
    set_num_threads(num_threads)

    translator = TranslatorWithSentencePiece(
        translation_model=translation_models, sentencepiece_model=sentencepiece_model
    )

    sentences = load_list_from_file(src_file)
    translations = translator.translate_sentences(sentences)
    dump_list_to_file(translations, output_file)


if __name__ == "__main__":
    translate_actions()

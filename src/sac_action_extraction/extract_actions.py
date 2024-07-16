"""Script to extract actions from a response string using predefined models."""

from pathlib import Path
from paragraph2actions.default_converters import default_action_converters
from paragraph2actions.paragraph_translator import ParagraphTranslator
from paragraph2actions.readable_converter import ReadableConverter

from typing import List

from .sac_converters import default_sac_converters

def extract_actions_from_string(response: str, models: List[Path] = [Path("/data/share/saccrow/models/sac.pt")], sentencepiece_model: Path = Path("/data/share/saccrow/models/sp_model.model")) -> dict:
    """Extract actions from a response string using predefined models.

    Args:
        response (str): The input text to process.
        models (list[Path]): List of Paths to translation model files.
        sentencepiece_model (Path): Path to the SentencePiece model file.

    Returns:
        dict: Extracted action objects.
    """
    # Combine default SAC converters with other default converters
    single_action_converters = default_sac_converters() + default_action_converters()
    converter = ReadableConverter(single_action_converters=single_action_converters)

    # Initialize the Paragraph Translator
    paragraph_translator = ParagraphTranslator(
        translation_model=[str(m) for m in models],
        sentencepiece_model=str(sentencepiece_model),
        action_string_converter=converter,
    )

    # Process the input response
    try:
        result = paragraph_translator.extract_paragraph(response)
        return {
            "actions": result.actions,
            "actions_readable": converter.actions_to_string(result.actions)
        }
    except Exception as e:
        return {"error": str(e)}

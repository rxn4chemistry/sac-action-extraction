from paragraph2actions.default_converters import default_action_converters
from paragraph2actions.readable_converter import ReadableConverter

from .sac_converters import default_sac_converters


def load_converter() -> ReadableConverter:
    single_action_converters = default_sac_converters() + default_action_converters()
    return ReadableConverter(single_action_converters=single_action_converters)

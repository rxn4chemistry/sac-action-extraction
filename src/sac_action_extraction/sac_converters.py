from typing import List

from paragraph2actions.conversion_utils import SingleActionConverter
from paragraph2actions.default_converters import (
    Parameters,
    action_with_parameters,
    default_action_converters,
    repetition_from_text,
    repetition_to_text,
)

from .sac_actions import (
    ALD,
    Centrifugate,
    Concentrate,
    Copolymerize,
    DrySolution,
    Grind,
    Impregnate,
    Leach,
    PostTreatment,
    Precipitate,
    Pyrolize,
    Reflux,
    Stir,
    SynthesisMethod,
    SynthesisProduct,
    SynthesisVariant,
    Transfer,
    Wait,
    Wash,
)

ald = action_with_parameters(
    ALD,
    [
        ("metal", "metal"),
        ("precursor", "precursor"),
        ("support", "support"),
        ("deposition_temperature", "deposited at"),
        ("flux_pressure", "fluxed under"),
        ("flux_temperature", "fluxed at"),
    ],
)

centrifugate = action_with_parameters(
    Centrifugate,
    [
        ("rpm", "rpm"),
        ("duration", "for"),
        ("temperature", "at"),
    ],
)

concentrate = action_with_parameters(
    Concentrate,
    [
        ("duration", "for"),
        ("temperature", "at"),
    ],
)

copolymerize = action_with_parameters(
    Copolymerize,
    [
        ("precursor", "from"),
        ("gas", "under"),
        ("duration", "for"),
        ("temperature", "at"),
        ("ramp", "ramp"),
    ],
    Parameters("material", None),
)

dry_solution = action_with_parameters(
    DrySolution,
    [
        ("material", "over"),
        ("duration", "for"),
        ("temperature", "at"),
    ],
)

grind = action_with_parameters(
    Grind,
    [
        ("material", None),
        ("duration", "for"),
        ("temperature", "at"),
    ],
)

leach = action_with_parameters(
    Leach,
    [
        ("duration", "for"),
        ("temperature", "at"),
    ],
    Parameters("material", None),
)

impregnate = action_with_parameters(
    Impregnate,
    [
        ("support_material", "on"),
        ("solvent", "in"),
        ("target_loading", "loading"),
    ],
    Parameters("material", None),
)

post_treatment = action_with_parameters(
    PostTreatment,
    [
        ("treatment_type", None),
        ("gas", "under"),
        ("duration", "for"),
        ("temperature", "at"),
        ("ramp", "ramp"),
    ],
)

precipitate = action_with_parameters(
    Precipitate,
    [
        ("support_material", "on"),
        ("solvent", "in"),
        ("target_concentration", "at"),
        ("ph", "with pH"),
        ("target_loading", "loading"),
    ],
    Parameters("material", None),
)

pyrolize = action_with_parameters(
    Pyrolize,
    [
        ("gas", "under"),
        ("duration", "for"),
        ("temperature", "at"),
        ("ramp", "ramp"),
    ],
    Parameters("material", None),
)

reflux = action_with_parameters(
    Reflux,
    [
        ("duration", "for"),
        ("temperature", "at"),
        ("atmosphere", "under"),
        ("dean_stark", "with Dean-Stark apparatus"),
    ],
)

stir = action_with_parameters(
    Stir,
    [
        ("duration", "for"),
        ("temperature", "at"),
        ("atmosphere", "under"),
        ("rpm", "rpm"),
    ],
)

synthesis_method = action_with_parameters(
    SynthesisMethod,
    [
        ("method", None),
    ],
)

synthesis_product = action_with_parameters(
    SynthesisProduct,
    [
        ("catalyst_name", None),
        ("target_loading", "loading"),
    ],
)

synthesis_variant = action_with_parameters(
    SynthesisVariant,
    [
        ("precursor", "with"),
        ("temperature", "at"),
    ],
    Parameters("target_metal", None),
)


transfer = action_with_parameters(
    Transfer,
    [
        ("vessel", "to"),
    ],
)

wait = action_with_parameters(
    Wait,
    [
        ("duration", "for"),
        ("temperature", "at"),
        ("atmosphere", "under"),
    ],
)

wash = action_with_parameters(
    Wash,
    [
        ("repetitions", None, repetition_to_text, repetition_from_text),
        ("duration", "for"),
        ("temperature", "at"),
    ],
    Parameters("material", "with"),
)


def default_sac_converters() -> List[SingleActionConverter]:
    """
    Get the string-action converters for the single-atom-catalyst synthesis actions.
    """
    action_converters: List[SingleActionConverter] = [
        ald,
        centrifugate,
        concentrate,
        copolymerize,
        dry_solution,
        grind,
        impregnate,
        leach,
        post_treatment,
        precipitate,
        pyrolize,
        reflux,
        stir,
        synthesis_method,
        synthesis_product,
        synthesis_variant,
        transfer,
        wait,
        wash,
    ]
    return action_converters


def default_sac_and_original_converters() -> List[SingleActionConverter]:
    """
    Get the string-action converters for the single-atom-catalyst synthesis
    actions as well as the original actions.
    """
    return default_sac_converters() + default_action_converters()

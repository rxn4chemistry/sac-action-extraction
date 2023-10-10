from typing import List

from action_sequences.paragraph_to_actions.paragraph_translator import (
    ParagraphTranslator,
)
from rxn_actions.actions import Add, MakeSolution, Yield
from rxn_actions.sac_actions import PostTreatment, SynthesisProduct

# SAME AS BEFORE
paragraphs: List[str] = []

# SAME AS BEFORE
paragraph_translator = ParagraphTranslator(
    translation_model="SAME AS BEFORE",
    sentencepiece_model="SAME AS BEFORE",
    action_string_converter="SAME AS BEFORE",  # type: ignore
)

n_stir = 0
temperatures: List[str] = []

add_actions: List[Add] = []
makesolution_actions: List[MakeSolution] = []
yield_actions: List[Yield] = []
synthesis_product_actions: List[SynthesisProduct] = []
posttreatment_actions: List[PostTreatment] = []

for paragraph in paragraphs:
    try:
        result = paragraph_translator.extract_paragraph(paragraph)
        for action in result.actions:
            if isinstance(action, Add):
                add_actions.append(action)
            if isinstance(action, MakeSolution):
                makesolution_actions.append(action)
            if isinstance(action, PostTreatment):
                posttreatment_actions.append(action)
            if isinstance(action, Yield):
                yield_actions.append(action)
            if isinstance(action, SynthesisProduct):
                synthesis_product_actions.append(action)
    except Exception as e:
        print("Error")
        print(e)

# Analysis
compound_names: List[str] = []
quantities: List[str] = []
products: List[str] = []
treatment_types: List[str] = []

for action in add_actions:
    compound_names.append(action.material.name)
    quantities.extend(action.material.quantity)
for action in makesolution_actions:
    for material in action.materials:
        compound_names.append(material.name)
        quantities.extend(material.quantity)
for action in yield_actions:
    products.append(action.material.name)
for action in synthesis_product_actions:
    products.append(action.catalyst_name)
for action in posttreatment_actions:
    treatment_types.append(action.treatment_type)

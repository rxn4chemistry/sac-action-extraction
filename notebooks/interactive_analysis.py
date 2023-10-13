import collections
from typing import Counter

from paragraph2actions.paragraph_translator import ParagraphTranslator
from paragraph2actions.sentence_splitting.cde_splitter import CdeSplitter
from paragraph2actions.utils import extract_compound_names, extract_temperatures
from rxn.utilities.logging import setup_console_logger

from sac_action_extraction.sac_actions import Stir
from sac_action_extraction.utils import load_converter

setup_console_logger()

model = "sac.pt"
sp_model = "sp_model.model"

converter = load_converter()

paragraph_translator = ParagraphTranslator(
    translation_model=model,
    sentencepiece_model=sp_model,
    sentence_splitter=CdeSplitter(),
    action_string_converter=converter,
)

# Two paragraph examples, the first one from an SAC procedure, the second one from an organic procedure.

paragraphs = [
    "In a typical procedure, nickel nitrate hexahydrate (3.48 g) and sodium citrate (4.71 g) were dissolved in 400 mL water to form clear solution A. In the meantime, potassium hexacyanocobaltate (III) (2.66 g) was dissolved into water (200 mL) to form clear solution B. Then, solutions A and B were mixed under magnetic stirring. Stirring was stopped after combining the component solutions. After 24 h, the solid was collected by centrifugation, washed with water and ethanol, and then dried at room temperature. Then, the dried sample was annealed at 500 °C in NH3 for 1 h with a slow heating rate of 2°C min−1 , and immersed in 5 M H2SO4 solution and stirred at 80°C for 24 h to form CoNi-SAs/NC hollow cubes."
    "A solution of ((1S,2S)-1-{[(4'-methoxymethyl-biphenyl-4-yl)-(2-pyridin-2-yl-cyclopropanecarbonyl)-amino]-methyl}-2-methyl-butyl)-carbamic acid tert-butyl ester (25 mg, 0.045 mmol) and dichloromethane (4 mL) was treated with a solution of HCl in dioxane (4 N, 0.5 mL) and the resulting reaction mixture was maintained at room temperature for 12 h. The reaction was then concentrated to dryness to afford (1R,2R)-2-pyridin-2-yl-cyclopropanecarboxylic acid ((2S,3S)-2-amino-3-methylpentyl)-(4'-methoxymethyl-biphenyl-4-yl)-amide (18 mg, 95% yield) as a white solid."
]

# We will count the number of Stir actions, and collect all the temperatures and compound names

n_stir = 0
temperatures = []
materials: Counter[str] = collections.Counter()

for paragraph in paragraphs:
    try:
        result = paragraph_translator.extract_paragraph(paragraph)
        for action in result.actions:
            if isinstance(action, Stir):
                n_stir += 1
        materials.update(extract_compound_names(result.actions, ignore_sln=True))
        temperatures.extend(extract_temperatures(result.actions))
    except Exception as e:
        print("Error")
        print(e)


print("Results")
print(f"{n_stir} Stir actions")
print("temperatures:", temperatures)
print("materials:", materials.most_common())

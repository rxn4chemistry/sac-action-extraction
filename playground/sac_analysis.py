import collections
from pathlib import Path
from typing import Counter

from paragraph2actions.default_converters import default_action_converters
from paragraph2actions.paragraph_translator import ParagraphTranslator
from paragraph2actions.utils import extract_compound_names, extract_temperatures

from .sac_actions import Stir
from .sac_converters import default_sac_converters
from .utils import load_converter

init_dir = Path("/Users/ava/Documents/client_projects/ace_eth/20220902_retraining")
model = init_dir / "model_lr_0.20_step_9000.pt"
sp_model = init_dir / "original_models" / "sp_model.model"

single_action_converters = default_sac_converters() + default_action_converters()
converter = load_converter()

paragraph_translator = ParagraphTranslator(
    translation_model=str(model),
    sentencepiece_model=str(sp_model),
    action_string_converter=converter,
)

sentences = [
    "The obtained sample was denoted as NiSA-N-CNTs.",
    "The precursor was placed in a crucible with a cover and annealed at 900 °C with a rate of 3 °C min−1 for 2 hr in Ar atmosphere.",
    "P-ACN  was obtained  by  the  photo-reduction  method.",
    "All catalysts were prepared by a facile adsorption method.",
    "Before characterization and test, all samples were reduced in an atmosphere of 10% H2/He at 200 oC for 0.5 hours.",
    "Bare reduced graphene oxide (rGO) was used as the control sample for parallel comparisons.",
    "After reacted for 12 h, a dark green Ni/SiO2@PDA was obtained.",
    "The metal oxides were removed by immersing the sample in the HF (20 wt%) solution at 60 °C.",
    "The typical synthesis process of Rh@S1-H was as follows: Firstly, 13 g TPAOH solution and 23.3 and 54.4 mg aluminum isopropoxide were added into 15 g of deionized water.",
    "After stirring  12  h,  the  water  was  removed  by  freeze-drying  to  obtain  the  powder.",
    "In a  typical  synthesis, 100 mg  ZIF67 powders  were  mixed  with  50  ml ethanol by  ultra-sonification for  30 min to form  a homogeneous  dispersion.",
    "Glucose  (0.5  g)  and  dicyandiamide  (2  g)  were  dissolved  in  30  mL  of  deionized  water, and  the  resulting  solution  then  heated  in  an  oil-bath  at  78  oC  under  magnetic  stirring until  the  solution  became  transparent.",
    "HAuCl4·4H2O and H2PtCl6·6H2O were dissolved in 15 mL water, and then added into the CeO2 solution dropwise.",
    "In a  typical  synthesis,  50  mg  PPy  powders  were  mixed with 50  ml ethanol  by ultra-sonification  for  30  min to form  a  homogeneous  dispersion.",
    "The HEB was obtained after deprotection of HEB-TMS by tetrabutyl ammonium fluoride (TBAF) and used immediately.",
    "A certain amount of BP2000 and Zn(OAC)2・2H2O with 30 mL 6 M HNO3 were added to round-bottom flask by ultrasonicating the mixture for 2-3 min, and stirred during refluxing at 80 °C for several hours, then dried up using rotary evaporator at 60 °C.",
]

n_stir = 0
temperatures = []
materials: Counter[str] = collections.Counter()

for sentence in sentences:
    try:
        result = paragraph_translator.extract_paragraph(sentence)
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

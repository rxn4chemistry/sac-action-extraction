# paragraph to actions on aCe@ETH data

import os
from pathlib import Path

import pandas as pd
from paragraph2actions.paragraph_translator import ParagraphTranslator

from .utils import load_converter

rxn_box_dir = Path(os.environ["IBM_BOX_DIR"])
p2a_model_dir = (
    rxn_box_dir / "Data" / "paragraph2actions" / "nat_comms_revision" / "models"
)
p2a_model_dir = Path("/data") / "tmp" / "aCe"

df: pd.DataFrame = pd.read_csv("/data/tmp/aCe/data.csv")

organic_models = [p2a_model_dir / f"augmented-{i}.pt" for i in range(1, 4)]
sp_model = p2a_model_dir / "sp_model.model"

init_dir = Path("/Users/ava/Documents/client_projects/ace_eth/20220902_retraining")
sac_model = init_dir / "model_lr_0.20_step_9000.pt"

converter = load_converter()

organic_paragraph_translator = ParagraphTranslator(
    translation_model=[str(model) for model in organic_models],
    sentencepiece_model=str(sp_model),
)
sac_paragraph_translator = ParagraphTranslator(
    translation_model=[str(sac_model)],
    sentencepiece_model=str(sp_model),
    action_string_converter=converter,
)

paragraphs = df["Paragraph"].tolist()

output_jsonl = True

count_identical = 0
count_non_identical = 0

with open("/tmp/annotations.out", "wt") as f:
    for p in paragraphs:
        organic_extracted = organic_paragraph_translator.extract_paragraph(p)
        sac_extracted = sac_paragraph_translator.extract_paragraph(p)
        for organic_sentence, sac_sentence in zip(
            organic_extracted.sentences, sac_extracted.sentences
        ):
            if not organic_sentence.text == sac_sentence.text:
                f.write(f"Difference sentence! {sac_sentence.text}\n")
                continue

            org_actions = converter.actions_to_string(organic_sentence.actions)
            sac_actions = converter.actions_to_string(sac_sentence.actions)
            if org_actions != sac_actions:
                f.write(f'Difference for "{organic_sentence.text}":\n')
                f.write(f'Org model: "{org_actions}":\n')
                f.write(f'SAC model: "{sac_actions}":\n\n')
                count_non_identical += 1
            else:
                count_identical += 1

print(f"{count_identical} sentences had identical actions.")
print(f"{count_non_identical} sentences had different actions.")

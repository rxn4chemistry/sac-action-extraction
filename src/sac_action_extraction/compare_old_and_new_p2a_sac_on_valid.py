# paragraph to actions on aCe@ETH data

import os
from pathlib import Path

from action_sequences.paragraph_to_actions.paragraph_translator import (
    ParagraphTranslator,
)
from rxn.utilities.files import iterate_lines_from_file
from tqdm import tqdm

from .utils import load_converter

rxn_box_dir = Path(os.environ["IBM_BOX_DIR"])
p2a_model_dir = (
    rxn_box_dir / "Data" / "paragraph2actions" / "nat_comms_revision" / "models"
)
p2a_model_dir = Path("/data") / "tmp" / "aCe"

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

src_file = init_dir / "20220811_annotations" / "splits" / "src-valid.txt"
tgt_file = init_dir / "20220811_annotations" / "splits" / "tgt-valid.txt"

src = iterate_lines_from_file(src_file)
tgt = iterate_lines_from_file(tgt_file)

output_jsonl = True

count_identical = 0
count_identical_with_gt = 0
count_non_identical = 0
org_correct = 0
sac_correct = 0

with open("/tmp/annotations.out", "wt") as f:
    for src_line, tgt_line in tqdm(zip(src, tgt)):
        organic_extracted = organic_paragraph_translator.extract_paragraph(src_line)
        sac_extracted = sac_paragraph_translator.extract_paragraph(src_line)

        org_actions = converter.actions_to_string(organic_extracted.actions)
        sac_actions = converter.actions_to_string(sac_extracted.actions)
        if org_actions == tgt_line:
            org_correct += 1
        if sac_actions == tgt_line:
            sac_correct += 1
        if sac_actions == org_actions:
            count_identical += 1

        if org_actions == sac_actions == tgt_line:
            count_non_identical += 1
        else:
            count_identical_with_gt += 1
            f.write(f'Difference for "{src_line}":\n')
            f.write(f'Annotation: "{tgt_line}":\n')
            f.write(f'Org model: "{org_actions}":\n')
            f.write(f'SAC model: "{sac_actions}":\n\n')

print(f"{count_identical_with_gt} sentences had identical actions with gt.")
print(f"{count_non_identical} sentences had different actions.")
print(f"{count_identical} sentences were extracted the same way.")
print(f"{org_correct} sentences are correct with organic model.")
print(f"{sac_correct} sentences are correct with SAC model.")

import csv
import re
from pathlib import Path
from typing import Iterator, Optional, Tuple

from rxn.utilities.files import PathLike, iterate_lines_from_file
from rxn.utilities.regex import capturing, integer_number_regex, real_number_regex

training_logs = "/Users/ava/Documents/client_projects/ace_eth/20230814_plot_convergence/20220905_training_logs.txt"


def get_step(line: str) -> Optional[int]:
    rgx = "Step " + capturing(integer_number_regex) + r"/" + integer_number_regex + ";"
    match = re.search(rgx, line)
    if match is None:
        return None
    return int(match.group(1))


def get_accuracy(line: str) -> Optional[float]:
    rgx = "Validation accuracy: " + capturing(real_number_regex)
    match = re.search(rgx, line)
    if match is None:
        return None
    return float(match.group(1))


def process(filename: PathLike, keep_multiple: int = 1) -> Iterator[Tuple[int, float]]:
    lines = iterate_lines_from_file(filename)

    step = -1
    for line in lines:
        step_match = get_step(line)
        accuracy_match = get_accuracy(line)

        if step_match is not None:
            step = step_match

        if accuracy_match is not None:
            if step % keep_multiple == 0:
                yield step, accuracy_match


expected = list(range(1000, 21000, 1000))

results_dir = Path(
    "/Users/ava/Documents/client_projects/ace_eth/20230815_plot_convergence_many"
)
files = [
    "augmented_025.txt",
    "augmented_050.txt",
    "augmented_075.txt",
    "augmented_100.txt",
    "nonaugmented_025.txt",
    "nonaugmented_050.txt",
    "nonaugmented_075.txt",
    "nonaugmented_100.txt",
]

full_paths = [results_dir / file for file in files]
names = [file[:-4] for file in files]

results = [list(process(fp, 1000)) for fp in full_paths]

with open(results_dir / "all.csv", "w") as f:
    writer = csv.writer(f)

    # header
    writer.writerow(["Finetuning step"] + names)

    for i in range(20):
        steps = [result[i][0] for result in results]
        accuracies = [result[i][1] for result in results]
        if len(set(steps)) != 1:
            raise ValueError(f"Problem! Several values for step: {steps}")
        writer.writerow([steps[0]] + accuracies)

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import attr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from paragraph2actions.actions import Action, InvalidAction, NoAction
from paragraph2actions.conversion_utils import ActionStringConversionError
from paragraph2actions.converter_interface import ActionStringConverter
from paragraph2actions.readable_converter import ReadableConverter
from rxn.utilities.files import load_list_from_file
from rxn.utilities.logging import setup_console_logger

from sac_action_extraction.utils import load_converter

# %matplotlib inline

setup_console_logger()
random.seed(42)

# Define file locations

metrics_dir = Path("/Users/ava/Documents/client_projects/ace_eth/20230217_metrics")
data_dir = metrics_dir / "data"
preds_dir = metrics_dir / "preds"

# ### Load data from files
#
# Careful, since both have not been processed identically...
# No InvalidAction in the rule-based data, for instance.
# Also, make sure that when including InvalidAction, it is the only action for a sentence (to make it consistent).

# NB: 'test' refers to the annotation test set
ace_sentences = load_list_from_file(data_dir / "ace" / "src-test.txt")
org_sentences = load_list_from_file(data_dir / "organic" / "src-test.txt")
gt_ace_as = load_list_from_file(data_dir / "ace" / "tgt-test.txt")
gt_org_as = load_list_from_file(data_dir / "organic" / "tgt-test.txt")

gt_all_ace_as = (
    load_list_from_file(data_dir / "ace" / "tgt-train.txt")
    + load_list_from_file(data_dir / "ace" / "tgt-valid.txt")
    + load_list_from_file(data_dir / "ace" / "tgt-test.txt")
)
gt_all_org_as = (
    load_list_from_file(data_dir / "organic" / "tgt-train.txt")
    + load_list_from_file(data_dir / "organic" / "tgt-valid.txt")
    + load_list_from_file(data_dir / "organic" / "tgt-test.txt")
)

ace_pred_as = load_list_from_file(preds_dir / "ace" / "ace-test.txt")
org_pred_as = load_list_from_file(preds_dir / "ace" / "organic-test.txt")

# Create action type counters


# +
def get_one_action_list(
    action_string: str, converter: ActionStringConverter
) -> List[Action]:
    try:
        action_list = converter.string_to_actions(action_string)
    except ActionStringConversionError:
        return [InvalidAction()]

    # Replace empty by NoAction()
    action_list = [NoAction()] if not action_list else action_list
    # If list has InvalidAction, remove all the other actions
    action_list = (
        [InvalidAction()]
        if any(isinstance(a, InvalidAction) for a in action_list)
        else action_list
    )

    return action_list


def get_action_lists(action_strings: List[str]) -> List[List[Action]]:
    converter = load_converter()
    return [
        get_one_action_list(action_string, converter)
        for action_string in action_strings
    ]


def get_normalized_action_counter(action_strings: List[str]) -> Counter:
    action_lists = get_action_lists(action_strings)
    action_types = [
        action.action_name for action_list in action_lists for action in action_list
    ]
    counter = Counter(action_types)
    total = sum(counter.values())
    for key in counter:
        counter[key] /= total  # type: ignore
    return counter


# +
gt_ace_action_counter = get_normalized_action_counter(gt_ace_as)
gt_org_action_counter = get_normalized_action_counter(gt_org_as)
gt_all_ace_action_counter = get_normalized_action_counter(gt_all_ace_as)
gt_all_org_action_counter = get_normalized_action_counter(gt_all_org_as)
ace_pred_action_counter = get_normalized_action_counter(ace_pred_as)
org_pred_action_counter = get_normalized_action_counter(org_pred_as)

gt_ace_actions_lists = get_action_lists(gt_ace_as)
gt_org_actions_lists = get_action_lists(gt_org_as)
x_gt_ace_action_lenghts = [len(s) for s in gt_ace_actions_lists]
x_gt_org_action_lenghts = [len(s) for s in gt_org_actions_lists]

# -

# Determine action labels in order of frequency when averaged over counters


def normalize(array) -> List[float]:
    total = sum(array)
    return [v / total for v in array]


def get_normalized_bar_heights(counter: Counter, labels: List[str]) -> List[float]:
    bar_height = [counter[label] for label in labels]
    return normalize(bar_height)


# +
def get_labels(*counters: Counter) -> List[str]:
    total_counter: Counter = sum(counters, Counter())
    labels: List[str] = [t[0] for t in total_counter.most_common()]
    return labels


all_action_labels = get_labels(
    gt_ace_action_counter, ace_pred_action_counter, org_pred_action_counter
)

# -


def get_histogram_data(
    labels: List[str], *counters: Counter
) -> Tuple[List[float], ...]:
    """
    Get the histogram data comparing the frequencies of two sets of actions.
    The values are returned in order of averaged frequency

    Returns: A tuple of 1. the list of labels, followed by 2.-x.: one list of values per given counter
    """
    bar_heights = [get_normalized_bar_heights(counter, labels) for counter in counters]
    return tuple(bar_heights)


# ### Plot distribution of actions

# SAC vs organic annotated dataset (not predictions)

# +
fig, ax = plt.subplots(1, figsize=(15, 7))

light_green = "#66c2a5"
light_blue = "#8da0cb"
light_orange = "#fc8d62"

action_labels = get_labels(gt_all_ace_action_counter)
action_labels.extend(l for l in all_action_labels if l not in action_labels)
ace_bar_height, org_bar_height = get_histogram_data(
    action_labels,
    gt_all_ace_action_counter,
    gt_all_org_action_counter,
)
ind = np.arange(len(action_labels))  # the x locations for the groups

ax.margins(x=0.02)
ax.bar(ind - 0.22, ace_bar_height, width=0.22, color=light_green)
ax.bar(ind - 0.0, org_bar_height, width=0.22, color=light_orange)
# ax.bar(ind + 0.22, test_bar_height, width=0.2, color=light_blue)
ax.set_ylabel("Frequency")

ax.set_title("Action distribution in annotated sentences (all splits)")
ax.set_xticks(ind, minor=False)
ax.set_xticklabels(action_labels, rotation=90)
ax.legend(
    labels=["Hand-annotated (SAC paragraphs)", "Hand-annotated (organic paragraphs)"]
)

plt.tight_layout()
plt.savefig("/tmp/ace_vs_org_annotations.pdf")
# -

# Annotation vs ACE model vs organic model, on the SAC annotated data

# +
fig, ax = plt.subplots(1, figsize=(15, 7))

light_green = "#66c2a5"
light_blue = "#8da0cb"
light_orange = "#fc8d62"

action_labels = get_labels(gt_ace_action_counter)
action_labels.extend(l for l in all_action_labels if l not in action_labels)
gt_bar_height, ace_bar_height, org_bar_height = get_histogram_data(
    action_labels,
    gt_ace_action_counter,
    ace_pred_action_counter,
    org_pred_action_counter,
)
ind = np.arange(len(action_labels))  # the x locations for the groups

ax.margins(x=0.02)
ax.bar(ind - 0.22, gt_bar_height, width=0.22, color=light_green)
ax.bar(ind - 0.0, ace_bar_height, width=0.22, color=light_orange)
ax.bar(ind + 0.22, org_bar_height, width=0.2, color=light_blue)
ax.set_ylabel("Frequency")

ax.set_title("Action distribution in SAC sentences (test split)")
ax.set_xticks(ind, minor=False)
ax.set_xticklabels(action_labels, rotation=90)
ax.legend(labels=["Hand-annotated", "SAC model", "organic model"])

plt.tight_layout()
plt.savefig("/tmp/predicted_actions.pdf")
# -

raise SystemExit("STOP HERE")

# Rule-based on pistachio vs annotation

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# First plot
action_labels = get_labels(rb_pistachio_action_counter)
action_labels.extend(l for l in all_action_labels if l not in action_labels)
rb_bar_height, an_bar_height = get_histogram_data(
    action_labels, rb_pistachio_action_counter, rb_annotation_action_counter
)
ind = np.arange(len(action_labels))  # the x locations for the groups

ax1.margins(x=0.02)
ax1.bar(ind - 0.175, rb_bar_height, width=0.35, color="#8da0cb")
ax1.bar(ind + 0.175, an_bar_height, width=0.35, color="#fc8d62")
ax1.set_xticks(ind, minor=False)
ax1.set_xticklabels(action_labels, rotation=90)
ax1.set_ylabel("Frequency")
ax1.legend(labels=["Rule-based on Pistachio", "Rule-based on annotation dataset"])


# Second plot
action_labels = get_labels(gt_annotation_action_counter)
action_labels.extend(l for l in all_action_labels if l not in action_labels)
an_bar_height, test_bar_height = get_histogram_data(
    action_labels, gt_annotation_action_counter, gt_ace_action_counter
)

ind = np.arange(len(action_labels))  # the x locations for the groups
ax2.margins(x=0.02)
ax2.bar(ind - 0.175, an_bar_height, width=0.35, color="#fc8d62")
ax2.bar(ind + 0.175, test_bar_height, width=0.35, color="#66c2a5")
ax2.set_xticks(ind, minor=False)
ax2.set_xticklabels(action_labels, rotation=90)
ax2.set_ylabel("Frequency")
ax2.legend(labels=["Hand-annotated, all splits", "Hand-annotated, test split"])

plt.tight_layout()
plt.savefig("/tmp/dataset_action_frequencies.pdf")

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# character length
n_bins = 50
x_pistachio = [len(s) for s in pistachio_sentences]
x_annotations = [len(s) for s in annotation_sentences]
kwargs = dict(histtype="stepfilled", alpha=0.3, density=True, bins=40, ec="k")
ax1.hist(
    x_pistachio,
    n_bins,
    histtype="bar",
    alpha=0.75,
    density=True,
    label=["Pistachio sentences"],
    range=(0, 600),
    color=["#8da0cb"],
)
ax1.hist(
    x_annotations,
    n_bins,
    histtype="bar",
    alpha=0.75,
    density=True,
    label=["Annotation sentences"],
    range=(0, 600),
    color=["#fc8d62"],
)
ax1.legend()
ax1.set_ylabel("Frequency")
# ax1.axes.get_yaxis().set_visible(False)
ax1.set_xlabel("Number of characters")

# action length
n_bins = 9
ax2.hist(
    x_rb_action_lenghts,
    bins=np.arange(n_bins + 1) + 0.6,
    histtype="bar",
    alpha=0.75,
    density=True,
    label=["Rule-based model for Pistachio sentences"],
    color=["#8da0cb"],
    rwidth=0.6,
)
ax2.hist(
    x_gt_ace_action_lenghts,
    bins=np.arange(n_bins + 1) + 0.4,
    histtype="bar",
    alpha=0.75,
    density=True,
    label=["Ground truth for annotation sentences"],
    color=["#fc8d62"],
    rwidth=0.6,
)
# plt.gca().axes.get_yaxis().set_visible(False)
plt.gcf().set_dpi(130)
ax2.legend()
ax2.set_xlabel("Number of actions")
ax2.set_ylabel("Frequency")
ax2.set_xticks(np.arange(1, n_bins + 1))

plt.tight_layout()
plt.savefig("/tmp/dataset_stats.pdf")
# -

# ### Panel combining all four subplots

# +
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))

light_blue = "#8da0cb"
light_orange = "#fc8d62"

# character length
n_bins = 50
x_pistachio = [len(s) for s in pistachio_sentences]
x_annotations = [len(s) for s in annotation_sentences]
kwargs = dict(histtype="stepfilled", alpha=0.3, density=True, bins=40, ec="k")
ax1.hist(
    x_pistachio,
    n_bins,
    histtype="bar",
    alpha=0.7,
    density=True,
    label=["Pistachio sentences"],
    range=(0, 600),
    color=[light_blue],
)
ax1.hist(
    x_annotations,
    n_bins,
    histtype="bar",
    alpha=0.7,
    density=True,
    label=["Annotation sentences"],
    range=(0, 600),
    color=[light_orange],
)
ax1.legend()
ax1.set_ylabel("Frequency")
# ax1.axes.get_yaxis().set_visible(False)
ax1.set_xlabel("Number of characters")

# action length
n_bins = 9
ax2.hist(
    x_rb_action_lenghts,
    bins=np.arange(n_bins + 1) + 0.6,
    histtype="bar",
    alpha=0.75,
    density=True,
    label=["Rule-based model for Pistachio sentences"],
    color=[light_blue],
    rwidth=0.6,
)
ax2.hist(
    x_gt_ace_action_lenghts,
    bins=np.arange(n_bins + 1) + 0.4,
    histtype="bar",
    alpha=0.75,
    density=True,
    label=["Ground truth for annotation sentences"],
    color=[light_orange],
    rwidth=0.6,
)
# plt.gca().axes.get_yaxis().set_visible(False)
plt.gcf().set_dpi(130)
ax2.legend()
ax2.set_xlabel("Number of actions")
ax2.set_ylabel("Frequency")
ax2.set_xticks(np.arange(1, n_bins + 1))

# First plot
action_labels = get_labels(rb_pistachio_action_counter)
action_labels.extend(l for l in all_action_labels if l not in action_labels)
rb_bar_height, an_bar_height = get_histogram_data(
    action_labels, rb_pistachio_action_counter, rb_annotation_action_counter
)
ind = np.arange(len(action_labels))  # the x locations for the groups

ax3.margins(x=0.02)
ax3.bar(ind - 0.175, rb_bar_height, width=0.35, color=light_blue)
ax3.bar(ind + 0.175, an_bar_height, width=0.35, color=light_orange)
ax3.set_xticks(ind, minor=False)
ax3.set_xticklabels(action_labels, rotation=90)
ax3.set_ylabel("Frequency")
ax3.legend(labels=["Rule-based on Pistachio", "Rule-based on annotation dataset"])

# Second plot
action_labels = get_labels(gt_annotation_action_counter)
action_labels.extend(l for l in all_action_labels if l not in action_labels)
an_bar_height, test_bar_height = get_histogram_data(
    action_labels, gt_annotation_action_counter, gt_ace_action_counter
)

ind = np.arange(len(action_labels))  # the x locations for the groups
ax4.margins(x=0.02)
ax4.bar(ind - 0.175, an_bar_height, width=0.35, color=light_blue)
ax4.bar(ind + 0.175, test_bar_height, width=0.35, color=light_orange)
ax4.set_xticks(ind, minor=False)
ax4.set_xticklabels(action_labels, rotation=90)
ax4.set_ylabel("Frequency")
ax4.legend(labels=["Hand-annotated, all splits", "Hand-annotated, test split"])

for ax, label in zip((ax1, ax2, ax3, ax4), ("a", "b", "c", "d")):
    ax.text(
        -0.05,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="right",
    )

plt.tight_layout(pad=2)
plt.savefig("/tmp/panel.pdf")
# -

# ### Compare actions

actions_truth = get_action_lists(gt_ace_as)
actions_pred = get_action_lists(ml_test_as)


@attr.s(auto_attribs=True)
class ActionTypeStatistics:
    """
    Collects the counts for a given action type.
    """

    total_pred: int = 0
    total_truth: int = 0
    type_match: int = 0
    perfect_matches: int = 0
    only_in_pred: int = 0
    only_in_truth: int = 0


# Compute the statistics. Basically makes counters of exact matches and type matches for each sentence, and derives the statistics from there.

# +
stats: Dict[str, ActionTypeStatistics] = {
    action_type: ActionTypeStatistics() for action_type in action_labels
}
errors = []


def action_counter_to_string(counter: Counter) -> str:
    """Converts an action counter to a string representation.

    Example output: Stir, Add, Filter, CollectLayer (2), Wash (3)
    """
    separate_action_strings = []
    for action_name, count in counter.items():
        action_string = action_name
        if count != 1:
            action_string = f"{count} x {action_string}"
        separate_action_strings.append(action_string)
    return ", ".join(separate_action_strings)


# list of the predictions: action type(s) -> action type(s)
matshow_preds = []

for truth_sequence, pred_sequence in zip(actions_truth, actions_pred):
    converter = ReadableConverter()
    truth_actions = [converter.action_to_string(a) for a in truth_sequence]
    pred_actions = [converter.action_to_string(a) for a in pred_sequence]
    truth_types = [a.action_name for a in truth_sequence]
    pred_types = [a.action_name for a in pred_sequence]

    truth_string_counter = Counter(truth_actions)
    pred_string_counter = Counter(pred_actions)
    exact_intersection = truth_string_counter & pred_string_counter

    exact_intersection_as_type: Counter = Counter()
    for key, value in exact_intersection.items():
        exact_intersection_as_type[converter.string_to_action(key).action_name] += value

    truth_type_counter = Counter(truth_types)
    pred_type_counter = Counter(pred_types)
    type_intersection = truth_type_counter & pred_type_counter

    partial_intersection = type_intersection - exact_intersection_as_type

    remaining_truth = truth_type_counter - type_intersection
    remaining_pred = pred_type_counter - type_intersection

    for key, value in truth_type_counter.items():
        stats[key].total_truth += value
    for key, value in pred_type_counter.items():
        stats[key].total_pred += value
    for key, value in exact_intersection_as_type.items():
        stats[key].perfect_matches += value
    for key, value in type_intersection.items():
        stats[key].type_match += value
        for _ in range(value):
            matshow_preds.append((key, key))
    for key, value in remaining_pred.items():
        stats[key].only_in_pred += value
    for key, value in remaining_truth.items():
        stats[key].only_in_truth += value

    if remaining_truth or remaining_pred:
        errors.append(
            str(dict(remaining_truth)) + " predicted as " + str(dict(remaining_pred))
        )
        matshow_preds.append(
            (
                action_counter_to_string(remaining_truth),
                action_counter_to_string(remaining_pred),
            )
        )
# -

# Present the statistics from a DataFrame

# +
data: Dict[str, List[str]] = defaultdict(list)
for key, action_stats in stats.items():
    data["action_type"].append(key)
    for col, count in attr.asdict(action_stats).items():
        data[col].append(count)

df = pd.DataFrame.from_dict(data)
print(df)
# -

# Print the errors, most common first

error_counter = Counter(errors)
for mc in error_counter.most_common():
    print(f"{mc[1]} times: {mc[0]}")

print(matshow_preds[:15])

# +
gt_labels_counter = Counter(x[0] for x in matshow_preds)
gt_labels = [x[0] for x in gt_labels_counter.most_common()]

# for the predictions, take the same order and then add the missing ones
pred_labels_counter = Counter(x[1] for x in matshow_preds)
pred_labels = [l for l in gt_labels if l in pred_labels_counter]
for key, _ in pred_labels_counter.most_common():
    if key not in pred_labels:
        pred_labels.append(key)

# place the empty label at the end
gt_labels.remove("")
pred_labels.remove("")
gt_labels.append("")
pred_labels.append("")
# -

heat_values = np.zeros((len(gt_labels), len(pred_labels)), dtype=int)
for gt, pred in matshow_preds:
    gt_index = gt_labels.index(gt)
    pred_index = pred_labels.index(pred)
    heat_values[gt_index, pred_index] += 1

# Rename the last label, which would otherwise just be the empty string
gt_labels[-1] = "(no action)"
pred_labels[-1] = "(no action)"

df_heat = pd.DataFrame(heat_values, index=gt_labels, columns=pred_labels)
plt.figure(figsize=(13, 10))
sns.heatmap(
    df_heat,
    annot=False,
    cmap="Blues",
    vmin=0,
    vmax=10,
    xticklabels=True,
    yticklabels=True,
)
# Add grid lines with the additional args to heatmap: linewidths=0.005, linecolor='black'
plt.tight_layout()
plt.savefig("/tmp/action_prediction_matrix.pdf")

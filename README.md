# Extraction of actions from experimental procedures of single-atom catalysts (SACs)

This repository contains the code for [Language models and protocol standardization guidelines for accelerating synthesis planning in heterogeneous catalysis](https://todo.todo).

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Model use, evaluation, and comparison](#model-use-evaluation-and-comparison)

# Overview

This repository contains code to train models for the extraction of actions from experimental procedures for single-atom catalysts. 
It builds on top of models for extracting actions from organic procedures, as described in [this publication](https://doi.org/10.1038/s41467-020-17266-6) and available in [this Git repository](https://github.com/rxn4chemistry/paragraph2actions).

This repository contains the following:

* Definition and handling of synthesis actions related to single-atom catalysts.
* Code for preparing and transforming the data.
* Training and usage of a transformer-based model.

A trained model can be freely used online at https://huggingface.co/spaces/rxn4chemistry/synthesis-protocol-extraction.


# Quick run
A docker image can be run like this:

> docker run -d -p 3212:5000 doncamilom/sac_action_extract:latest

Which exposes an endpoint directly to the paragraphs2actions function.

To use in python, use like this:

```python
import json
import requests

def call_api(text):
    response = requests.post(
        "http://liacpc11.epfl.ch:3212/api/v1/actions",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"inputs": text})
    )
    return response.json()

string = "A solution of ((1S,2S)-1-{[(4-methoxymethyl-biphenyl-4-yl)-(2-pyridin-2-yl-cyclopropanecarbonyl)-amino]-methyl}-2-methyl-butyl)-carbamic acid tert-butyl ester (25 mg, 0.045 mmol) and dichloromethane (4 mL) was treated with a solution of HCl in dioxane (4 N, 0.5 mL) and the resulting reaction mixture was maintained at room temperature for 12 h. The reaction was then concentrated to dryness to afford (1R,2R)-2-pyridin-2-yl-cyclopropanecarboxylic acid ((2S,3S)-2-amino-3-methylpentyl)-(4'-methoxymethyl-biphenyl-4-yl)-amide (18 mg, 95% yield) as a white solid."

actions = call_api(string)
print(actions)

```


# System Requirements

## Hardware requirements
The code can run on any standard computer.
It is recommended to run the training scripts in a GPU-enabled environment.

## Software requirements
### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS: Ventura (13.6)
+ Linux: Ubuntu 20.04.3

### Python
A Python version of 3.7 or 3.8 is recommended.
The Python package dependencies are listed in [`setup.cfg`](./setup.cfg).

# Installation guide

To use the package, we recommended to create a dedicated `conda` or `venv` environment:
```bash
# Conda
conda create -n sac-action-extraction python=3.8
conda activate sac-action-extraction

# venv
python3.8 -m venv myenv
source myenv/bin/activate
```

The package can be installed with:
```bash
pip install -e .[dev]
```
The installation should not take more than a few minutes.


# Data preparation

## Annotation

The starting point are a set of annotated pairs of sentences and associated actions.
To make the execution of the scripts below easier, you should set the environment variable:
```bash
export ANNOTATON_DIR=/path/to/annotations
```

This directory should contain the annotations in one of two formats:
1) (preferred) A "JSONL" file called `annotated.json` with one entry per line in the following format: `{"sentence": "After two hours at 100 \u00b0C, 1 ml of water was added.", "actions": "WAIT for two hours at 100 \u00b0C; ADD water (1 ml)."}`.
2) `sentences.txt` with one sentence per line (f.i. `After two hours at 100 °C, 1 ml of water was added.`), and `actions.txt` with the associated actions, in Python format (f.i. `[Wait(duration='two hours', temperature='100 °C'), Add(material=Chemical(name='water', quantity=['1 ml']), dropwise=False, temperature=None, atmosphere=None, duration=None)]`)

## Create splits

Create splits with 10% in the validation set and 10% in the test set.
```bash
export ANNOTATION_SPLITS=/path/to/annotation/splits
sac-create-annotation-splits -a $ANNOTATION_DIR -o $ANNOTATION_SPLITS -v 0.1 -t 0.1
```
This will create the files `src-test.txt`, `src-train.txt`, `src-valid.txt`, `tgt-test.txt`, `tgt-train.txt`, and `tgt-valid.txt` in `$ANNOTATION_SPLITS`.

## Concatenating datasets

If you generated multiple datasets, each split already with the above command, you can combine them in the following way (with the environment variables adequately set):
```bash
sac-concatenate-annotations --dir1 $SPLITS_1 --dir2 $SPLITS_2 --combined $ANNOTATION_SPLITS
```

## Augmenting the datasets

The datasets are augmented in a way similar to the one described [here](https://github.com/rxn4chemistry/paragraph2actions#data-augmentation).

We provide a script to augment the train split (the validation and test splits are not changed).
It will replace at random, in all the sentences and associated actions, the compound names, quantities, durations and temperatures.
The script therefore requires many such compound names, quantities, durations and temperatures to be provided, so that random ones can be picked for augmentation.

The script requires the following files in a directory (f.i. specified by the environment variable `VALUE_LISTS_DIR`) that contains:
* `compound_names.txt` with one compound name per line: `triisopropyl borate`, `N-Boc-l-amino-l-hydroxymethylcyclopropane`, `2-tributylstannylpyrazine`, `Compound M`, `NH2NH2`, etc.
* `quantities.txt` with one quantity per line: `278mg`, `1.080 mL`, `3.88 mL`, `005 g`, `29.5 mmol`, `1.752 moles`, etc.
* `durations.txt` with one duration per line: `131 d`, `0.5 h`, `1½ h`, `about 7.5 hours`, etc.
* `temperatures.txt` with one temperature per line: `75-85°C`, `-78 Celsius`, `0 ∼ 5 °C`, `133° C`, `approximately -15° C`, etc.
 
```bash
export ANNOTATION_SPLITS_AUGMENTED=/path/to/augmented/annotation/splits
sac-augment-annotations -v $VALUE_LISTS_DIR -d $ANNOTATION_SPLITS -o $ANNOTATION_SPLITS_AUGMENTED
```
This will create the augmented files in the directory `$ANNOTATION_SPLITS_AUGMENTED`.

# Training

Note: the training procedure is similar as in the [`paragraph2actions` repository](https://github.com/rxn4chemistry/paragraph2actions#training-the-transformer-model-for-action-extraction), which contains more details on the individual steps.

We now assume that you followed the steps above (or equivalent ones), and that your dataset is present in `DATA_DIR`, with the following files:
```
src-test.txt    src-train.txt   src-valid.txt   tgt-test.txt    tgt-train.txt   tgt-valid.txt
```
If you performed augmentation, the training files may be named `src-train-augmented.txt` and `tgt-train-augmented`. 
Make sure to adapt the commands below if needed.

We also assume that you start from a pretrained model `pretrained_model.pt` (see [here](https://github.com/rxn4chemistry/paragraph2actions#training-the-transformer-model-for-action-extraction) for instructions to create one).

## Tokenization

```bash
paragraph2actions-tokenize -m $DATA_DIR/sp_model.model -i $DATA_DIR/src-train.txt -o $DATA_DIR/tok-src-train.txt
paragraph2actions-tokenize -m $DATA_DIR/sp_model.model -i $DATA_DIR/src-valid.txt -o $DATA_DIR/tok-src-valid.txt
paragraph2actions-tokenize -m $DATA_DIR/sp_model.model -i $DATA_DIR/tgt-train.txt -o $DATA_DIR/tok-tgt-train.txt
paragraph2actions-tokenize -m $DATA_DIR/sp_model.model -i $DATA_DIR/tgt-valid.txt -o $DATA_DIR/tok-tgt-valid.txt
```
(see [these instructions](https://github.com/rxn4chemistry/paragraph2actions#subword-tokenization) to obtain `sp_model.model`).

## OpenNMT preprocessing

```bash
onmt_preprocess \
  -train_src $DATA_DIR/tok-src-train.txt -train_tgt $DATA_DIR/tok-tgt-train.txt \
  -valid_src $DATA_DIR/tok-src-valid.txt -valid_tgt $DATA_DIR/tok-tgt-valid.txt \
  -save_data $DATA_DIR/onmt_preprocessed -src_seq_length 300 -tgt_seq_length 300 \
  -src_vocab_size 16000 -tgt_vocab_size 16000 -share_vocab
```

## Model finetuning

```
export LEARNING_RATE=0.20
onmt_train \
  -data $DATA_DIR/onmt_preprocessed  \
  -train_from pretrained_model.pt \
  -save_model $DATA_DIR/models  \
  -seed 42 -save_checkpoint_steps 1000 -keep_checkpoint 40 \
  -train_steps 30000 -param_init 0  -param_init_glorot -max_generator_batches 32 \
  -batch_size 4096 -batch_type tokens -normalization tokens -max_grad_norm 0  -accum_count 4 \
  -optim adam -adam_beta1 0.9 -adam_beta2 0.998 -decay_method noam -warmup_steps 8000  \
  -learning_rate $LEARNING_RATE -label_smoothing 0.0 -report_every 200  -valid_batch_size 512 \
  -layers 4 -rnn_size 256 -word_vec_size 256 -encoder_type transformer -decoder_type transformer \
  -dropout 0.1 -position_encoding -share_embeddings -valid_steps 200 \
  -global_attention general -global_attention_function softmax -self_attn_type scaled-dot \
  -heads 8 -transformer_ff 2048 -reset_optim all -gpu_ranks 0
```
This training script will take on the order of one hour to execute on one GPU, and will create model checkpoints in `$DATA_DIR/models`.
The learning rate and other parameters may be tuned; the values given here provided the best validation accuracy.


# Model use, evaluation, and comparison

The trained models compared in the publication can be downloaded from [this link](https://huggingface.co/spaces/rxn4chemistry/synthesis-protocol-extraction/tree/main).

## Action extraction

The following command allows you to make predictions with all three models (you may need to adapt the paths).
```bash
paragraph2actions-translate -s src-test.txt -o pred-ace.txt        -p sp_model.model -t sac.pt
paragraph2actions-translate -s src-test.txt -o pred-organic.txt    -p sp_model.model -t organic-1.pt -t organic-2.pt -t organic-3.pt
paragraph2actions-translate -s src-test.txt -o pred-pretrained.txt -p sp_model.model -t pretrained.pt
```

It is also possible to do interactive predictions from the command-line with the following:
```bash
paragraph2actions-translate -s sp_model.model -m sac.pt
```

## Metrics

To compute the metrics and produce a CSV comparing the models, execute the following:
```bash
sac-metrics-grid -g tgt-test.txt -p pred-ace.txt -p pred-organic.txt -p pred-pretrained.txt -o metrics.csv
```
It will create the file `metrics.csv` with the metrics.

## Analysis of extracted actions

[`interactive_analysis.py`](./notebooks/interactive_analysis.py) illustrates, in an interactive manner, how one can gain insight into the model predictions.
To run it as an IPython notebook, run the following command to create `interactive_analysis.ipynb` (after installing `jupytext` from PyPI):
```bash
jupytext --set-format py,ipynb notebooks/interactive_analysis.py
```

# Motivation in Large Language Models

This repository contains the code used to run experiments and reproduce the analyses.

## Setup

Create the environment:

```
conda env create -f env.yml
conda activate motivation-env
```

## Credentials

API credentials are loaded by `credits/credits.py` from JSON files located in the `credits/` directory.

Before running experiments, place the required credential files in `credits/` (e.g., API keys for the relevant providers).

## Running Experiments

Run the experiments:

```
python run_experiments.py
```

Then merge the outputs:

```
python merge_per_model.py
```

Merging is required before running the analysis notebooks.

## Analysis

The analysis notebooks are located in the repository root:

* `analyze_choice.ipynb`
* `analyze_human_exp.ipynb`
* `analyze_manipulations.ipynb`
* `analyze_none.ipynb`
* `analyze_text.ipynb`

Run these notebooks to reproduce the analyses.

## Experiment Logs

The exact experiment logs from the authors’ runs are not included in this public repository.

Experiment logs are available to reviewers and will be released publicly upon publication.

## Notes

Running experiments requires API access and may incur costs.

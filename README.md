# DeMMO

DeMMO is an interpretable multi-task learning framework for longitudinal,
multi-outcome, and cross-disease analysis of digital mobility outcomes (DMOs).
It combines fused temporal regularisation, sparse group selection, and
automatic relation learning across clinical prediction objectives.

This repository provides the implementation code for DeMMO.

## Repository contents

- `demmo/`: core optimiser and Mobilise-D data-loading utilities.
- `scripts/run_demmo_two_stage.py`: two-stage hyperparameter selection and
  held-out evaluation.
- `scripts/run_demmo_variants.py`: two-stage evaluation of the non-convex
  DeMMO-var1 and DeMMO-var2 extensions.
- `scripts/run_hyperparameter_sensitivity.py`: one-at-a-time sensitivity
  analysis.
- `scripts/run_longitudinal_stability_selection.py`: balanced half-sample
  longitudinal stability selection.
- `tests/`: unit tests for proximal operators and alternating optimisation.

## Data availability and privacy

The Mobilise-D data are not redistributed in this repository. Researchers
must obtain authorised access from the official data provider and comply with
the applicable data-use agreement.

Never commit raw data, participant identifiers, participant-level splits,
clinical outcomes, or individual predictions. The included `.gitignore`
blocks the filenames produced for these sensitive artifacts by the experiment
scripts. Review all generated files before changing the repository visibility
or creating a release.

The expected local dataset layout is:

```text
<dataset-root>/
└── outcome/
    ├── PD_dataset.csv
    ├── MS_dataset.csv
    └── PFF_dataset.csv
```

The loader expects 24 complete weekly DMO columns, a participant identifier,
a visit indicator, and the clinical outcome columns used by the four
prediction objectives. Data remain local and are never downloaded or uploaded
by the code.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Two-stage DeMMO experiment

The first stage selects the fused-Lasso, element-wise Lasso, and group-Lasso
penalties with relation learning disabled. The second stage fixes those values
and selects the two relation-learning penalties.

```bash
python scripts/run_demmo_two_stage.py /path/to/Mobilise-D \
  --output-dir outputs/seed_42 \
  --seed 42 \
  --fl-grid 0.001,0.01,0.1,1 \
  --l-grid 0.001,0.01,0.1,1 \
  --g-grid 0.001,0.01,0.1,1 \
  --r-grid 0.001,0.01,0.1 \
  --a-grid 0.001,0.01,0.1
```

Run the experiment with seeds 42--46 to reproduce the five matched
participant-level splits. All visits from one participant remain in the same
split, and preprocessing statistics are estimated from training data only.

## Non-convex extensions

DeMMO-var1 and DeMMO-var2 replace the convex sparsity terms with reweighted
non-convex penalties designed to reduce coefficient shrinkage bias. Each
majorisation iteration solves a reweighted convex subproblem, making these
extensions substantially more computationally demanding than DeMMO. They are
initialised from the convex DeMMO solution and use the same two-stage model
selection protocol and participant splits.

```bash
python scripts/run_demmo_variants.py /path/to/Mobilise-D \
  --output-dir outputs/var1_seed_42 \
  --variant var1 \
  --seed 42 \
  --convex-parameters FL L G R A
```

Replace `var1` with `var2` for the second extension. `FL L G R A` are the five
DeMMO hyperparameters selected on the corresponding participant split.

## Hyperparameter sensitivity

```bash
python scripts/run_hyperparameter_sensitivity.py /path/to/Mobilise-D \
  --output-dir outputs/hyperparameter_sensitivity \
  --seed 42
```

## Longitudinal stability selection

The reported analysis uses ten randomly selected hyperparameter settings and
ten independent half-sample repetitions for each setting. Only training
participants are sampled.

```bash
python scripts/run_longitudinal_stability_selection.py \
  --dataset-root /path/to/Mobilise-D \
  --output-dir outputs/longitudinal_stability \
  --hyperparameter-settings 10 \
  --repeats-per-setting 10
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Citation

The manuscript citation and preprint DOI will be added after the arXiv record
is available.

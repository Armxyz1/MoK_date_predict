# MoK_date_predict

Monsoon over Kerala prediction is a climate-machine-learning project developed as a BTech final-year project. The repository uses the ERA5 reanalysis dataset to study whether historical geophysical variables can be used to learn useful representations of the pre-monsoon atmosphere and then predict the date of monsoon onset over Kerala.

The project follows a two-stage workflow:

1. A self-supervised BYOL model learns robust spatiotemporal representations from climate inputs.
2. A downstream regression model uses the extracted features to predict the onset date, with both standard evaluation and Leave-One-Year-Out (LOYO) evaluation.

The repository also contains explainability utilities for inference visualisation, attribution maps, and feature extraction from trained checkpoints.

## Project Objective

The aim of this work is to build a data-driven model for Monsoon over Kerala prediction that can:

- learn from multi-channel climate data,
- capture temporal evolution across successive timesteps,
- produce interpretable regression outputs for onset-date prediction,
- evaluate generalisation under both regular validation and year-wise LOYO testing,
- support model interpretability through attribution-based visual analysis.

## Repository Overview

The main workflow is organised around three stages:

1. Data preprocessing and representation learning.
2. Regression training and evaluation.
3. Explainability, inference visualisation, and feature extraction.

### Core Directories

- `src/training/scripts`: Training and evaluation entry points.
- `src/explainability`: Inference, attribution, and feature extraction scripts.
- `config`: Model and data configuration files.
- `checkpoints`: Saved model weights.
- `features`: Precomputed feature tensors used by the regressors.
- `results`: Evaluation tables, plots, and summary artifacts.
- `outputs`: Sample-level explainability figures.

## Training and Evaluation Scripts

The `src/training/scripts` directory contains the main files used to run the project.

### BYOL Representation Learning

`src/training/scripts/train_byol.py`

This script trains the BYOL backbone used to learn climate representations in a self-supervised manner. It reads the project configuration, creates the dataloaders, applies the expected normalisation pipeline, and saves the learned representation model.

The BYOL stage is used to produce transferable features that are later consumed by the regression models.

### Standard Regression Evaluation

`src/training/scripts/train_delta_regressor_cross_val.py`

This script trains the delta sequence regressor on precomputed features using the normal train/validation/test split. It performs hyperparameter search, selects the best configuration, and writes the resulting prediction tables and metrics to `results/tables`.

Typical outputs include:

- train, validation, and test prediction CSV files,
- a grid-search summary,
- the best hyperparameter JSON file,
- a saved checkpoint for the best regressor,
- per-timestep skill score tables.

### Leave-One-Year-Out Evaluation

`src/training/scripts/train_delta_regressor_cross_val_loyo.py`

This script performs LOYO evaluation, where each test year is held out sequentially and the model is re-trained using all previous years as additional training data. This is the most important evaluation mode for measuring year-wise generalisation in a climate forecasting setting.

Typical outputs include:

- sequential test prediction CSV files,
- summary JSON and CSV reports,
- top-ranked hyperparameter trials,
- best-trial selection summaries.

## Explainability and Feature Extraction

The `src/explainability` directory contains the utilities used for model interpretation and post-training analysis.

### Full-Model Explainability

`src/explainability/full_model_explain.py`

This is the main inference-and-explainability entry point. It loads the trained BYOL backbone and the regressor checkpoint, runs inference on the test set, and produces side-by-side visualisations of the input channels and their attribution maps.

The generated images are saved under `outputs/sample_XXXXX/` as per-timestep figures.

### Combined Model and Attribution Wrapper

`src/explainability/combined_model.py`

This file defines the combined inference pipeline and the model wrapper used for attribution. It contains the backbone, the delta regressor, the feature preparation logic, and the attribution support used by the explainability script.

### Feature Extraction

`src/explainability/full_model_extract.py`

This script is intended for extracting features from the trained model pipeline and preparing intermediate outputs for further analysis.

## Data and Artifacts

### Configuration

- `config/model_config.yml` is the main configuration file used by the training and evaluation scripts.
- `config/model_config_new_45.yml` is an additional configuration variant used for the BYOL feature pipeline.

### Saved Features

The downstream regressors expect feature tensors stored under:

- `features/<feature_subdir>/train_features.pt`
- `features/<feature_subdir>/val_features.pt`
- `features/<feature_subdir>/test_features.pt`

For this repository, the main feature directory is `features/MoK_byol_new_transformer_45/`.

### Checkpoints

Model checkpoints are stored in `checkpoints/`, including the BYOL backbone checkpoint and the best regressor checkpoint.

The trained checkpoints can be found accessed from the following links:
- [BYOL Backbone Checkpoint](https://drive.google.com/file/d/1KjylcWLOfzDaFo3313Mce6MOeoF9VoGw/view?usp=sharing)
- [Best Regressor Checkpoint](https://drive.google.com/file/d/1__p0R0aSFRTnAKHwi-9EPwKlRfYflE5e/view?usp=sharing)

### Results

The `results/` directory contains:

- prediction CSV files,
- hyperparameter search summaries,
- LOYO evaluation summaries,
- skill score tables,
- plots and figures generated during analysis.

### Explainability Outputs

The `outputs/` directory contains sample-wise attribution figures saved by the explainability pipeline.

## Recommended Workflow

1. Prepare the configuration in `config/model_config.yml`.
2. Train or load the BYOL checkpoint using `src/training/scripts/train_byol.py`.
3. Generate and store feature tensors in `features/MoK_byol_new_transformer_45/`.
4. Train the standard regressor with `train_delta_regressor_cross_val.py`.
5. Run LOYO evaluation with `train_delta_regressor_cross_val_loyo.py`.
6. Use `src/explainability/full_model_explain.py` to generate attribution maps and inference figures.

## Requirements

This project is implemented in Python and depends on common scientific and deep-learning libraries. The conda environment can be set up using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate MoK_date_predict
```

## Usage Notes

- Several scripts currently use project-specific absolute paths inside the source code. If you move the repository to a different location, update those paths before running the scripts.
- The regression scripts assume that feature files already exist in the `features/` directory.
- The explainability script expects trained BYOL and regressor checkpoints to be available in `checkpoints/`.
- The LOYO script is computationally more expensive than the standard train/validation/test run because it retrains across multiple year splits.
- For the complete methodology, experimental setup, and results discussion, refer to [MoK_paper.pdf](MoK_paper.pdf).

## Example Outputs

- `results/tables/*.csv` for evaluation summaries and predictions.
- `results/hyperparams/*.yml` for run configuration logs.
- `checkpoints/*.pth` for saved model weights.
- `outputs/sample_*/side_by_side_t*_all_channels.png` for attribution visualisations.

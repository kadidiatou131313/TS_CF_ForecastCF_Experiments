# forecastcf-univariate-ts-counterfactuals

Experimental evaluation of ForecastCF, a gradient-based counterfactual explanation method for univariate time series forecasting, originally proposed by Wang et al. (ICDM 2023).

Three sequential experiments (E0, E1, E2, E3) are conducted across four benchmark datasets and three deep learning architectures, investigating the reliability of the forecasting benchmark, the relationship between forecasting performance and counterfactual quality, the advantage of ForecastCF over naive baselines, and the sensitivity of counterfactual quality to the target tolerance width.

---

## Reference

Wang, Z., Miliou, I., Samsten, I., and Papapetrou, P. (2023). Counterfactual Explanations for Time Series Forecasting. In 2023 IEEE International Conference on Data Mining (ICDM), pp. 1391-1396. https://doi.org/10.1109/ICDM58522.2023.00180

Original implementation : https://github.com/zhendong3wang/counterfactual-explanations-for-forecasting

---

## Repository structure

```
forecastcf-univariate-ts-counterfactuals/
|
|-- ExperimentationsForcastCF.ipynb   Main experiment notebook (E0, E1, E2, E3)
|
|-- a2_forecastcf.py                  ForecastCF algorithm (PyTorch port)
|-- a2bis_forecastcf.py               ForecastCF + baseline methods (ShiftCF, NNCF)
|-- a3Metrics.py                      Counterfactual evaluation metrics
|-- forecast_e0.py                    Forecasting model wrappers (Darts)
|
|-- saved_models_e0/                  Pre-trained model checkpoints (.pkl + .ckpt)
|   |-- ETTh1_DLinear.pkl
|   |-- ETTh1_GRU.pkl
|   |-- ...
|
|-- Rapport_annexe_Kadidiatou_Diallo.pdf
|-- Projet_Kadidiatou_Diallo.pdf 
|-- README.md
|-- .gitignore
```

---

## Module descriptions

### a2_forecastcf.py

PyTorch port of the ForecastCF algorithm from Wang et al. (2023). The original implementation provided by the authors was built in TensorFlow/Keras. This version re-implements the full gradient-based optimization loop in PyTorch, using `torch.optim.Adam` and native autograd, so that it integrates directly with the Darts-based forecasting models used in these experiments (which expose internal PyTorch modules). The core logic follows Algorithm 1 of the paper : iterative Adam updates on the input series until all forecast timesteps fall within the target bounds, controlled by a masking-based margin loss.

### a2bis_forecastcf.py

Extends `a2_forecastcf.py` with two naive baseline methods used in E2.

`BaselineShiftCF` applies a fixed multiplicative shift to the input series. It requires no model access and serves as the simplest possible perturbation strategy.

`BaselineNNCF` retrieves the nearest neighbor from the training set whose corresponding model prediction is closest (Euclidean distance) to the center of the target interval. It requires a prior call to `.fit(X_train, Y_train)` to build the neighbor index.

### a3Metrics.py

Implements the four counterfactual evaluation metrics defined in Wang et al. (2023), adapted from their original NumPy/pandas formulation.

`validity_ratio` measures the proportion of forecast timesteps falling within the target bounds (Ratio in the paper).

`proximity_l2` measures the average L2 distance between the original input window and the counterfactual (Proximity in the paper).

`compactness_score` measures the proportion of input timesteps that remain unchanged within a tolerance (Compact in the paper).

`stepwise_validity_auc` measures the area under the curve of consecutive valid forecast steps (Step-AUC in the paper). This metric was adapted to handle both single-sample and batch inputs, and to use `np.trapezoid` (replacing the deprecated `np.trapz` in recent NumPy versions).

### forecast_e0.py

Contains the forecasting model wrapper classes used to train and evaluate models via the Darts library. `BaseDartsWrapper` handles data preparation and the fit/predict interface. `TorchDartsWrapper` adds PyTorch Lightning configuration with early stopping. Specialized subclasses (`DLinearDartsWrapper`, `RNNDartsWrapper`, `NBEATSDartsWrapper`) configure each architecture. The `build_base_configuration` factory function is the entry point used in the notebook.

---

## Datasets

All datasets are loaded directly from the Darts built-in dataset library (`darts.datasets`), requiring no manual download.

`ETTh1` (Electricity Transformer Temperature, hourly). Multivariate industrial time series with strong daily and weekly seasonality. Target : HUFL (high useful load). Chosen for its structured temporal patterns and wide use in forecasting benchmarks.

`Weather` (weather station, Germany). Multivariate meteorological series sampled at 10-minute intervals, containing 21 variables. Target : p (mbar), atmospheric pressure. Chosen for its high regularity, which provides a controlled setting where all models are expected to perform well.

`Sunspots` (monthly sunspot count, 1749-present). Univariate series with a well-known approximately 11-year cycle. Chosen as a long-range univariate benchmark with non-stationary periodicity.

`Electricity` (hourly electricity consumption, UCI). High-dimensional multivariate series with 370 clients. Target : MT_001 (first client). Chosen for its scale and real-world demand patterns.

These four datasets cover complementary properties : seasonality, regularity, long-range cycles, and high dimensionality, providing a diverse and representative benchmark.

---

## Forecasting architectures

Three architectures are evaluated, each representing a distinct family of deep forecasting models.

`DLinear` decomposes the input into trend and seasonal components and applies separate linear projections to each. It is a strong linear baseline that is competitive with Transformer-based models on many benchmarks despite its simplicity.

`GRU` (Gated Recurrent Unit) is a recurrent architecture that processes the input sequence step by step, maintaining a hidden state. It is well-suited to series with strong temporal dependencies and sequential dynamics.

`NBeats` (Neural Basis Expansion Analysis) stacks fully connected residual blocks without any recurrence or attention. It is interpretable by design and has demonstrated state-of-the-art performance in time series forecasting competitions.

Together, these three architectures cover linear decomposition, recurrent processing, and pure MLP-based approaches, ensuring that results are not specific to one modeling family.

---

## Notebook

`ExperimentationsForcastCF.ipynb` contains all four experiments in sequence. Models trained in E0 remain in memory and are reused in E1, E2, and E3 without reloading.

**E0. Reference benchmark.** Trains and evaluates all 12 (dataset, model) pairs (4 datasets x 3 architectures) using a 60/20/20 temporal split and StandardScaler fitted on the training set only. Pairs are retained for subsequent experiments if R² exceeds 0.50 on the test set. All 12 pairs pass this threshold (minimum R² = 0.636 for Electricity/DLinear).

**E1. Forecasting performance vs. counterfactual quality.** For each retained pair, 10 counterfactuals are generated from randomly sampled test windows. CF metrics are averaged per pair and Spearman correlation is computed against R². Result : no significant correlation is found across any metric (Validity: rho = -0.078, p = 0.809; Proximity : rho = -0.399, p = 0.199; Compactness : rho = 0.000, p = 1.000), indicating that forecasting performance does not predict counterfactual quality.

**E2. Comparison with baselines.** ForecastCF is compared to ShiftCF and NNCF on all 12 pairs, using 10 instances per pair. Wilcoxon signed-rank tests confirm that ForecastCF significantly outperforms both baselines on Validity, Proximity, and Compactness (p < 0.05 in all cases), at the cost of significantly higher generation time.

**E3. Sensitivity to target tolerance.** The target interval width is varied across three values (0.1 * std, 0.3 * std, 0.6 * std) with a fixed shift of 0.5 * std. A width of 0.3 * std is identified as the best practical compromise between validity and compactness.

---

## Saved models

The `saved_models_e0/` directory contains one `.pkl` file and one `.pkl.ckpt` file per (dataset, model) pair. The `.pkl` file is the serialized Darts model object. The `.ckpt` file is the underlying PyTorch Lightning checkpoint. Both are required to reload a model. They are provided so that E1, E2, and E3 can be reproduced without retraining.

---

## Requirements

```
Python >= 3.9

torch >= 2.0
darts >= 0.27
scikit-learn >= 1.3
numpy >= 1.24
pandas >= 2.0
matplotlib >= 3.7
scipy >= 1.11
pytorch-lightning >= 2.0
```

Install all dependencies :

```bash
pip install torch darts scikit-learn numpy pandas matplotlib scipy
```

---

## How to run

1. Clone the repository :

```bash
git clone https://github.com/kadidiatou31919/forecastcf-univariate-ts-counterfactuals
cd forecastcf-univariate-ts-counterfactuals
```

2. Install dependencies (see Requirements above).

3. Open the notebook :

```bash
jupyter notebook ExperimentationsForcastCF.ipynb
```

4. To skip retraining, run the cell that loads models from `saved_models_e0/` instead of the training loop. Both options are provided in the E0 section of the notebook.

5. Experiments E1, E2, and E3 depend on the `trained_models` and `datasets_bank` objects built in E0. Run all E0 cells before proceeding.

---

## Notes

The target bounds used in E1, E2, and E3 follow an adaptive formulation based on the standard deviation of each input window : lower bound = pred + shift * std, upper bound = pred + (shift + width) * std, with shift = 0.5 and width = 0.5 by default. This normalizes the constraint across datasets with different scales, consistent with the polynomial trend instantiation described in Wang et al. (2023).

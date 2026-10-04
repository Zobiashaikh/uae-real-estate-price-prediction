# UAE Real Estate Price Prediction
### CSCI323 Modern Artificial Intelligence — Spring 2026
**Institution:** University of Wollongong Dubai (UOWD)

**[▶ Watch the demo video](https://drive.google.com/file/d/1dnCS3V_wpCOOLm49tJE5ryPIe-u85heA/view?usp=sharing)** · **[View the presentation](Presentation.pdf)**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dataset](#dataset)
4. [Requirements](#requirements)
5. [Setup and Installation](#setup-and-installation)
6. [How to Run](#how-to-run)
7. [Model Summary](#model-summary)
8. [Feature Engineering](#feature-engineering)
9. [Reproducing Results](#reproducing-results)
10. [Code Quality Standards](#code-quality-standards)
11. [Team Contributions](#team-contributions)
12. [References](#references)

---

## Project Overview

This project addresses the problem of **residential real estate price prediction in the UAE**, built for a UAE-based real estate platform. Manual property valuation is time-consuming, inconsistent, and difficult to scale. This solution automates price estimation using supervised machine learning, enabling more accurate and data-driven pricing decisions.

**Business Problem:** Estimating the sale price of a UAE residential property based on its features (location, size, number of rooms, furnishing status, etc.).

**AI Approach:** Regression using ensemble models trained on 5,058 real UAE property listings from 2024, with the target variable log-transformed to handle price skewness.

**Best Model:** `HistGradientBoostingRegressor` (tuned)
- **Test R²:** 0.7836 (original AED scale)
- **Test RMSE:** 925,388 AED
- **Test MAE:** 579,614 AED

---

## Repository Structure

```
project/
│
├── CSCI323_PROJECT.ipynb       # Main training notebook (EDA → preprocessing → model training → evaluation → saving)
├── Prediction_Model.ipynb      # Interactive prediction interface (load model → user input → price output)
│
├── uae_real_estate_2024.csv    # Dataset (see Dataset section below)
├── predictor.pkl               # Trained model, saved by CSCI323_PROJECT.ipynb
├── Presentation.pdf            # Project presentation slides
│
└── README.md                   # This file
```

> **Note:** The trained model `predictor.pkl` is included, so `Prediction_Model.ipynb` works straight away. Running `CSCI323_PROJECT.ipynb` to completion regenerates it; see [Reproducing Results](#reproducing-results).

---

## Dataset

| Property | Details |
|---|---|
| **File** | `uae_real_estate_2024.csv` |
| **Rows** | 5,058 property listings |
| **Columns** | 12 raw columns |
| **Source** | UAE real estate listings (2024) |
| **Target** | `price` (AED, integer) — log-transformed during training |

### Raw Columns

| Column | Type | Description |
|---|---|---|
| `title` | str | Property listing title (dropped — unstructured text) |
| `displayAddress` | str | Area and city (e.g., `Hayyan, Sharjah`) |
| `bathrooms` | str | Number of bathrooms |
| `bedrooms` | str | Number of bedrooms |
| `addedOn` | str | Listing date (used to extract month) |
| `type` | str | Always `"Residential for Sale"` (dropped — constant) |
| `price` | int64 | Sale price in AED — **prediction target** |
| `verified` | bool | Whether the listing is verified |
| `priceDuration` | str | Always `"sell"` (dropped — constant) |
| `sizeMin` | str | Property size in sqft |
| `furnishing` | str | `NO`, `YES`, or `PARTLY` |
| `description` | str | Free-text description (dropped — unstructured text) |

### Missing Values (raw data)

| Column | Missing |
|---|---|
| `bathrooms` | 120 |
| `bedrooms` | 123 |
| `furnishing` | 2 |
| `description` | 1 |
| All others | 0 |

Missing values in `bathrooms` and `bedrooms` are imputed using the median strategy inside the model pipeline.

---

## Requirements

### Python Version
Python **3.10 or higher** is required. The project was developed and tested on Python **3.12.6**.

### Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Plotting and visualisation |
| `seaborn` | Statistical visualisation |
| `scikit-learn` | ML models, preprocessing, evaluation, pipelines |
| `catboost` | CatBoost gradient boosting model |
| `joblib` | Model serialisation (saving/loading `.pkl`) |

All dependencies are available on PyPI and can be installed with `pip` (see [Setup and Installation](#setup-and-installation)).

---

## Setup and Installation

### 1. Clone or Download the Repository

```bash
git clone https://github.com/Zobiashaikh/uae-real-estate-price-prediction.git
cd uae-real-estate-price-prediction
```

Or click **Code → Download ZIP** on GitHub and extract it.

### 2. (Recommended) Create a Virtual Environment

```bash
# Create the environment
python -m venv venv

# Activate it — Windows
venv\Scripts\activate

# Activate it — macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

**Option A — Install all at once:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost joblib
```

**Option B — Using the notebook's built-in install cells:**

Open `CSCI323_PROJECT.ipynb`, navigate to **Section 0.2**, uncomment the `!pip install` lines, and run that cell. Re-comment them afterwards.

```python
# Section 0.2 in CSCI323_PROJECT.ipynb
!pip install pandas numpy matplotlib seaborn -q
!pip install scikit-learn -q
!pip install catboost -q
```

### 4. Place the Dataset

Ensure `uae_real_estate_2024.csv` is in the **same directory** as the notebooks:

```
project/
├── CSCI323_PROJECT.ipynb
├── Prediction_Model.ipynb
└── uae_real_estate_2024.csv     ← place it here
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open either notebook from the Jupyter interface in your browser.

---

## How to Run

There are two notebooks with distinct purposes:

### `CSCI323_PROJECT.ipynb` — Training Pipeline

This notebook covers the full ML workflow from raw data to a saved model. Run all cells **in order**:

| Section | Description |
|---|---|
| **0 — Environment** | Import libraries, optionally install them, load the CSV |
| **1 — Initial Analysis** | Shape, data types, missing values, duplicates |
| **2 — Preprocessing** | Drop irrelevant columns, clean and encode features, extract month from date, parse `sizeMin`, one-hot encode `furnishing` |
| **3 — EDA** | Distribution plots, correlation heatmaps, price analysis by location and room count |
| **4 — Feature Engineering** | Derived ratio features (bathroom/bedroom ratio, size per bedroom, etc.) |
| **5 — Model Training & Tuning** | Train/val/test split, 5-fold CV, baseline and tuned versions of Ridge, Random Forest, HistGradientBoosting, CatBoost |
| **6 — Summary** | Final performance comparison table |
| **7 — Save Model** | Saves `predictor.pkl` to the project directory |

> **Total runtime:** approximately 14–15 minutes on a standard laptop (CatBoost and RandomizedSearchCV are compute-intensive).

### `Prediction_Model.ipynb` — Interactive Price Predictor

This notebook provides an interactive menu for predicting property prices.

**Prerequisite:** `predictor.pkl` must exist in the same directory (generated by running the training notebook above).

Run all cells. A menu will appear:

```
Menu
1. New Prediction
2. Edit Last Prediction
3. Exit

Select option:
```

**Input fields:**

| Field | Description | Format |
|---|---|---|
| Area and City | Location string | e.g., `Downtown Dubai, Dubai` |
| Bathrooms | Number of bathrooms | Integer ≥ 0 |
| Bedrooms | Number of bedrooms | Integer ≥ 0 |
| Verified | Listing verified? | `Y` or `N` |
| Size (sqft) | Property size | Integer ≥ 0 |
| Month Listed | Month the property was listed | Integer 1–12 |
| Furnished | Furnishing status | `Y` (Yes), `N` (No), or `P` (Partly) |

Enter `q` at any numeric prompt to quit. Press **Enter** with no input on option 2 to keep existing values.

**Example output:**

```
================================
Predicted Price: AED 2,450,000
================================
```

---

## Model Summary

Four regression models were trained and evaluated. The target variable (`price`) was log-transformed using `log1p` to reduce skewness; predictions are converted back with `expm1`.

| Model | Baseline CV R² (Log) | Tuned CV R² (Log) | Test R² (Original) |
|---|---|---|---|
| Ridge Regression | 0.7979 | 0.7996 | — |
| Random Forest | 0.8219 | 0.8289 | — |
| **HistGradientBoosting** | **0.8220** | **0.8330** | **0.7836** |
| CatBoost | 0.8269 | 0.8321 | — |

**Selected model:** `HistGradientBoostingRegressor` (best tuned CV R² with lowest overfitting risk)

The final model is saved as a full scikit-learn `Pipeline` object in `predictor.pkl`, which includes all preprocessing steps and the trained estimator.

---

## Feature Engineering

In addition to the cleaned raw features, the following interaction and ratio features were engineered:

| Feature | Formula | Rationale |
|---|---|---|
| `bathroom_bedroom_ratio` | `bathrooms / bedrooms` | Captures relative room balance |
| `size_per_bedroom` | `sizeMin / bedrooms` | Space efficiency per bedroom |
| `size_per_bathroom` | `sizeMin / bathrooms` | Space efficiency per bathroom |
| `bedrooms_per_size` | `bedrooms / sizeMin` | Room density |
| `bedrooms_bathrooms_interaction` | `bedrooms × bathrooms` | Joint room count signal |


The categorical column `displayAddress` (area and city) is encoded using **Target Encoding** inside the pipeline.

---

## Reproducing Results

To reproduce the results from scratch:

1. Complete all steps in [Setup and Installation](#setup-and-installation).
2. Open and run **`CSCI323_PROJECT.ipynb`** from top to bottom.
3. Confirm `predictor.pkl` has been created in the project directory.
4. Open and run **`Prediction_Model.ipynb`** to use the trained model.

All random operations use fixed seeds where applicable to ensure reproducibility. Results may vary slightly depending on OS, library versions, and hardware.

---

## Code Quality Standards

This project follows the code quality guidelines specified in the CSCI323 project description:

- **Meaningful names:** All variables and functions use descriptive names reflecting their purpose.
- **Comments:** Complex logic is explained inline; section headers are used throughout the notebooks.
- **Modular structure:** Code is separated into logically distinct functions (`load_model`, `build_input_dataframe`, `get_property_input`, `predict_price`).
- **No hardcoded paths:** File paths are relative; the model filename is defined as a variable.
- **Warnings suppressed cleanly:** `warnings.filterwarnings("ignore")` is used to keep output readable without hiding errors.

---

## Team Contributions

Developed as a team project for CSCI323 (Modern Artificial Intelligence) at the University of Wollongong Dubai. My role was data preprocessing.

---

## References

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). Association for Computing Machinery. https://doi.org/10.1145/2939672.2939785
- Kanchana1990. (2024). *UAE real estate 2024 dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/kanchana1990/uae-real-estate-2024-dataset
- McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61). Python in Science Conference. https://proceedings.scipy.org/articles/Majora-92bf1922-00a.pdf
- Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter*, 3(1), 27–32. https://doi.org/10.1145/507533.507538
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html
- Phan, T. D. (2018). Housing price prediction using machine learning algorithms: The case of Melbourne City, Australia. *In 2018 International Conference on Machine Learning and Data Engineering (iCMLDE)* (pp. 35–42). IEEE. https://doi.org/10.1109/iCMLDE.2018.00017
- Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. *In Advances in Neural Information Processing Systems* (Vol. 31, pp. 6638–6648). Curran Associates, Inc. https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf
- Rosen, S. (1974). Hedonic prices and implicit markets: Product differentiation in pure competition. *Journal of Political Economy*, 82(1), 34–55. http://www.jstor.org/stable/1830899
- Scikit-learn developers. (2025). HistGradientBoostingRegressor. *Scikit-learn documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
- Scikit-learn developers. (2025). RandomForestRegressor. *Scikit-learn documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- Scikit-learn developers. (2025). Ridge. *Scikit-learn documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Scikit-learn developers. (2025). TargetEncoder. *Scikit-learn documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html
- Yoo, S., Im, J., & Wagner, J. E. (2012). Variable selection for hedonic model using machine learning approaches: A case study in Onondaga County, *NY. Landscape and Urban Planning*, 107(3), 293–306. https://doi.org/10.1016/j.landurbplan.2012.06.009


---


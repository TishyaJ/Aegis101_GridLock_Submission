# Flipkart Gridlock Hackathon: Spatio-Temporal Travel Demand Forecasting

**Team:** Aegis101  
**Peak Public Leaderboard Score:** 89.13446  

## 📌 Overview
This repository contains the complete, review-safe machine learning pipeline for the Flipkart Gridlock hackathon. The objective is to forecast highly skewed, spatio-temporal traffic demand across various `geohash` locations using a robust gradient boosting ensemble.

Our approach specifically avoids target leakage and "lookup table" exploits present in public baselines, achieving a highly competitive score through pure feature engineering, mathematically sound cross-validation, and strategic ensembling.

---

## 🏗️ System Architecture
The pipeline is entirely self-contained within `Gridlock_Submission.ipynb` and executes in the following phases:

1. **Data Ingestion & Cleaning:** Reads `train.csv` and `test.csv`. Critically, it **purges Day 49** from the training set to prevent temporal target leakage into the test set.
2. **Feature Engineering:** Extracts cyclical time patterns, spatial topologies, and infrastructure flags.
3. **Imputation:** Uses Geohash-grouped spatial means for `Temperature` instead of a flat global median, preserving the 2nd most important signal in the dataset.
4. **Intra-Day 5-Fold Time-Series Validation:** Because the test set is an unseen future day, random K-Fold validation causes severe target leakage. We implemented an expanding-window `TimeSeriesSplit` across the 96 intra-day time slots of Day 48 to generate a highly robust local R² estimation.
5. **Full-Dataset Training & Ensembling:** Wipes the memory, retrains on 100% of Day 48 data, and blends predictions using a weighted ensemble of LightGBM, CatBoost, and XGBoost.

---

## 🔬 Feature Engineering
Our feature matrix was engineered to capture both the physical layout of the city and the behavioral patterns of the traffic:

* **Temporal Features:** 
  * Extracted `day`, `hour`, and `minute`.
  * Applied **Cyclical Encoding** (`sin_time`, `cos_time`) to help the trees understand that 23:45 and 00:00 are adjacent.
  * Extracted logical flags: `is_morning_peak` and `is_evening_peak`.
* **Spatial & Topological Features:**
  * Converted `lat` / `lon` into 3D Cartesian coordinates (`x`, `y`, `z`) to preserve spatial distances.
  * **Spatial Density:** Used a `KNeighborsRegressor` to map the density of nearby traffic hubs.
* **Infrastructure Flags:** 
  * Boolean flags for `LargeVehicles` and `Landmarks`.
  * Designed a `bottleneck_index` combining `RoadType` capacity and demand.

---

## 🚀 Models & Tools Used
We utilized an optimized weighted ensemble of the industry's fastest gradient boosting frameworks:

* **LightGBM (Weight: 0.40):** Extremely fast leaf-wise tree growth, excellent for handling the high cardinality of `geohash`.
* **CatBoost (Weight: 0.35):** Natively handles the remaining categorical infrastructure features without requiring heavy pre-processing.
* **XGBoost (Weight: 0.25):** Provides robust depth-wise regularization to prevent the ensemble from overfitting the heavily skewed target demand.
* **Optuna:** Used offline for dynamic hyperparameter search (tree depths, learning rates, bagging fractions) to perfectly optimize the models for the `MSE` objective.
* **Scikit-Learn:** Core pipeline utilities (`TimeSeriesSplit`, `KNeighborsRegressor`, `r2_score`).

---

## ⚙️ Reproducibility
To reproduce the 89.13 score:
1. Ensure `dataset/train.csv` and `dataset/test.csv` are placed in the root directory.
2. Run `Gridlock_Submission.ipynb` sequentially from top to bottom.
3. The resulting `submission.csv` will be perfectly formatted for the leaderboard.

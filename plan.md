# 📘 PLAN — Validation, Feature Selection & Hyperparameter Optimization

## 🎯 Goal
To understand and implement:
- data validation schemes (hold-out, k-fold, stratified, time-series)
- feature selection methods (Lasso, correlation, permutation, SHAP)
- hyperparameter tuning (GridSearch, RandomSearch, Optuna)

---

## ✅ 00_questions.ipynb — Theoretical Part

- [X] Explain **Leave-One-Out** (strengths and limitations)
- [X] Describe how **Grid Search**, **Random Search**, and **Bayesian Optimization** work
- [X] Classify **feature selection methods**
- [X] Explain how **Pearson** and **Chi2** work
- [X] Explain how **Lasso** performs feature selection
- [X] Explain **permutation importance**
- [X] Explore and summarize **SHAP** values

🧠 *Goal:* Build solid theoretical grounding before implementing methods.

---

## ✅ 01_preprocessing.ipynb — Data Preparation

- [X] Import required libraries  
- [X] Load dataset
- [X] Preprocess the `"Interest Level"` feature
- [X] Generate binary features:
  - `Elevator`, `HardwoodFloors`, `CatsAllowed`, `DogsAllowed`, `Doorman`, `Dishwasher`,  
    `NoFee`, `LaundryinBuilding`, `FitnessCenter`, `Pre-War`, `LaundryinUnit`,  
    `RoofDeck`, `OutdoorSpace`, `DiningRoom`, `HighSpeedInternet`,  
    `Balcony`, `SwimmingPool`, `LaundryInBuilding`, `NewConstruction`, `Terrace`
- [X] Combine all 23 features into final feature set
- [X] Define `X` and `y`

📊 *Goal:* Get a clean, consistent dataset ready for all further experiments.

---

## ✅ 02_splitting_methods.ipynb — Data Splitting Functions

- [X] Implement random 2-part split (`test_size`)
- [X] Implement random 3-part split (`validation_size`, `test_size`)
- [ ] Implement date-based 2-part split (`date_split`)
- [ ] Implement date-based 3-part split (`validation_date`, `test_date`)
- [ ] Make all splits **deterministic** using random seed
- [ ] Verify data proportions and non-overlapping sets

🔁 *Goal:* Create reproducible splitting schemes for fair model evaluation.

---

## ✅ 03_cross_validation.ipynb — Cross-Validation Schemes

- [ ] Implement **K-Fold**
- [ ] Implement **Grouped K-Fold**
- [ ] Implement **Stratified K-Fold**
- [ ] Implement **TimeSeries Split**
- [ ] Apply sklearn equivalents for comparison
- [ ] Compare distributions of target and features across folds
- [ ] Visualize fold splits
- [ ] Select the **best CV scheme** for this dataset and justify the choice

🧩 *Goal:* Understand differences between CV methods and their suitable use-cases.

---

## ✅ 04_feature_selection.ipynb — Feature Selection Methods

- [ ] Fit **Lasso Regression** with normalized features (60/20/20 split)
- [ ] Sort and select **top 10 features** by absolute coefficient values
- [ ] Implement **NaN ratio + correlation**-based feature filter
- [ ] Implement **Permutation Importance**
- [ ] Apply **SHAP** analysis for interpretability
- [ ] Compare feature sets by:
  - performance (MAE, RMSE, R²)
  - computation time
  - stability (variance of results)

🎯 *Goal:* Identify which selection methods produce the most robust and interpretable models.

---

## ✅ 05_hyperparameter_tuning.ipynb — Hyperparameter Optimization

- [ ] Implement **Grid Search** and **Random Search** for `ElasticNet` (alpha, l1_ratio)
- [ ] Find optimal combination of hyperparameters
- [ ] Implement **Optuna** optimization for the same model
- [ ] Compare results between search strategies:
  - accuracy / error metrics
  - search time
  - stability
- [ ] Apply Optuna with one of your cross-validation schemes (e.g., StratifiedKFold)

⚙️ *Goal:* Learn to automate and optimize model performance tuning efficiently.

---

## ✅ 06_summary.ipynb — Results & Comparison

- [ ] Create summary tables for:
  - MAE, RMSE, R² for all models and validation schemes
  - time and stability comparison
- [ ] Visualize results with bar charts / boxplots
- [ ] Discuss:
  - Which validation scheme was most reliable?
  - Which feature selection method gave best performance?
  - Which hyperparameter optimization approach was most efficient?
- [ ] Final conclusion:  
  → Best overall pipeline = **validation + feature selection + tuning method**

🏁 *Goal:* Consolidate all results and extract insights.

---

## 📂 src Modules (code reusability)

| File | Purpose |
|------|----------|
| `data_split.py` | Train/Test/Validation splitting functions |
| `cross_validation.py` | Custom CV implementations |
| `feature_selection.py` | Lasso, correlation, permutation, SHAP |
| `hyperopt.py` | Grid Search, Random Search, Optuna tuning |
| `metrics.py` | MAE, RMSE, R², and helper metrics |
| `utils.py` | Seed fixing, reproducibility, helper tools |

---

## 🧩 Optional Enhancements

- [ ] Add visualization of feature importance (barplots)
- [ ] Create a small summary report (`report.md`)
- [ ] Save best model and parameters to `/models/saved_models/`
- [ ] Automate metrics logging to `.csv` file

---

## 💬 Final Deliverables

- 6 Jupyter notebooks (clear, modular, with Markdown explanations)
- Full working codebase in `src/validation_feature_selection/`
- `plan.md` with completed checklist
- Updated `README.md` describing your results and findings

---

🪶 *Tip:* Keep notebooks short and focused.  
At the start of each notebook, import functions from your `src` package — this way, the code stays clean and professional.

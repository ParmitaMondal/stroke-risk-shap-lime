# Stroke Risk Prediction with SHAP & LIME

End-to-end **tabular** ML pipeline to predict stroke risk using **demographics + morphology**
features, with model interpretability via **SHAP** and **LIME**.

- Model: Gradient-boosted trees (XGBoost)
- Preprocessing: numeric imputation + scaling, categorical imputation + one-hot
- Metrics: AUC, accuracy, classification report
- Explainability:
  - **SHAP**: global (summary beeswarm & bar)
  - **LIME**: local instance-level HTML reports

## Data

Provide a CSV with a binary target column (default: `stroke`).

**Examples of columns** (you can use your own):
- **Demographics**: `age`, `sex`, `bmi`, `hypertension`, `diabetes`, `smoking_status`
- **Morphology**: `infarct_core_ml`, `penumbra_ml`, `mismatch_ratio`, `aspects`,
  `collateral_score`, `clot_length_mm`, `ica_stenosis_pct`, `m1_occlusion`, `time_to_treatment_min`
- **Target**: `stroke` (0/1)


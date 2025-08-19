import os
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from xgboost import XGBClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to input CSV")
    ap.add_argument("--target", type=str, default="stroke", help="Target column (binary 0/1)")
    ap.add_argument("--numeric_cols", type=str, default="", help="Comma-separated numeric column names")
    ap.add_argument("--categorical_cols", type=str, default="", help="Comma-separated categorical column names")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="outputs", help="Output folder for plots")
    return ap.parse_args()


def infer_feature_types(df, target, user_num, user_cat):
    if user_num or user_cat:
        numeric_cols = [c.strip() for c in user_num.split(",") if c.strip()]
        categorical_cols = [c.strip() for c in user_cat.split(",") if c.strip()]
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_cols, categorical_cols


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Target column '{args.target}' not found in CSV.")

    y = df[args.target].astype(int).values
    X = df.drop(columns=[args.target])
    num_cols, cat_cols = infer_feature_types(df, args.target, args.numeric_cols, args.categorical_cols)
    print("Numeric columns:", num_cols)
    print("Categorical columns:", cat_cols)

    # Demographics: age, sex, bmi, hypertension, diabetes, smoking_status
    # Morphology: infarct_core_ml, penumbra_ml, mismatch_ratio, aspects, collateral_score,
    #             clot_length_mm, ica_stenosis_pct, m1_occlusion, time_to_treatment_min

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
 
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
   
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    clf = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=args.seed,
        eval_metric="logloss",
        n_jobs=4,
    )

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    pipe.fit(
        X_tr, y_tr,
        clf__eval_set=[(preprocessor.fit_transform(X_val), y_val)],
        clf__early_stopping_rounds=50,
        clf__verbose=False
    )

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest AUC:  {auc:.4f}")
    print(f"Test ACC:  {acc:.4f}\n")
    print(classification_report(y_test, y_pred, digits=3))

    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC Curve (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "roc_curve.png"), dpi=180)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=180)
    plt.close()

    Xt_train = pipe.named_steps["preprocess"].transform(X_train)
    Xt_test = pipe.named_steps["preprocess"].transform(X_test)
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(Xt_test)

    shap.summary_plot(shap_values, Xt_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_summary_beeswarm.png"), dpi=180)
    plt.close()

    shap.summary_plot(shap_values, Xt_test, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "shap_summary_bar.png"), dpi=180)
    plt.close()

    class_names = ["no-stroke", "stroke"]
   
    lime_explainer = LimeTabularExplainer(
        training_data=np.asarray(Xt_train),
        feature_names=list(feature_names),
        class_names=class_names,
        mode="classification",
        discretize_continuous=False
    )

    model_only = pipe.named_steps["clf"].predict_proba
    for i in range(min(3, Xt_test.shape[0])):
        exp = lime_explainer.explain_instance(
            Xt_test[i],
            model_only,
            num_features=10,
        )
        html_path = os.path.join(args.out, f"lime_example_{i}.html")
        exp.save_to_file(html_path)
        print(f"Saved LIME explanation: {html_path}")

    print(f"\nSaved figures to: {os.path.abspath(args.out)}")
    print("Files:")
    for f in ["roc_curve.png", "confusion_matrix.png", "shap_summary_beeswarm.png", "shap_summary_bar.png"]:
        print("  -", os.path.join(args.out, f))

if __name__ == "__main__":
    main()

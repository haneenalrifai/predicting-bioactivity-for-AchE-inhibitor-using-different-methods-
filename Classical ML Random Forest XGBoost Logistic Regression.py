

# Full Code


# ================================================================
# Publication-Oriented AChE Inhibitor ML Pipeline
# ================================================================

# ================================================================
# IMPORTS
# ================================================================

import numpy as np
import pandas as pd
import random
import os
import joblib
import warnings

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_validate
)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ================================================================
# GLOBAL REPRODUCIBILITY
# ================================================================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# ================================================================
# LOAD DATA
# ================================================================

DATA_PATH = r"C:\Users\QBS-OM\OneDrive\Desktop\Videos\BIS Lectures\AIML\AIML assignment\AChE_IC50_clean_labeled_more_descriptors.csv"

required_cols = [
    "canonical_smiles",
    "activity_class",
    "Molecular Weight",
    "AlogP",
    "#RO5 Violations"
]

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)
df = df[required_cols].dropna().reset_index(drop=True)

print("\nDataset summary")
print(df["activity_class"].value_counts())

# ================================================================
# FINGERPRINT GENERATION
# ================================================================

FP_SIZE = 2048
RADIUS = 2

morgan_gen = GetMorganGenerator(radius=RADIUS, fpSize=FP_SIZE)


def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    fp = morgan_gen.GetFingerprint(mol)

    arr = np.zeros((FP_SIZE,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    return arr


print("\nGenerating Morgan fingerprints...")

X_fp = df["canonical_smiles"].apply(smiles_to_fp)

mask = X_fp.notnull()

X_fp = np.stack(X_fp[mask])

y = df.loc[mask, "activity_class"].reset_index(drop=True)

# ================================================================
# DESCRIPTORS
# ================================================================

DESC_COLS = [
    "Molecular Weight",
    "AlogP",
    "#RO5 Violations"
]

X_desc = df.loc[mask, DESC_COLS].values



label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================

X_fp_train, X_fp_test, X_desc_train, X_desc_test, y_train, y_test = train_test_split(
    X_fp,
    X_desc,
    y_encoded,
    test_size=0.20,
    stratify=y_encoded,
    random_state=SEED
)
# ================================================================
# DESCRIPTOR SCALING
# ================================================================

scaler = StandardScaler()

X_desc_train = scaler.fit_transform(X_desc_train)

X_desc_test = scaler.transform(X_desc_test) 
# ================================================================
# COMBINE FEATURES
# ================================================================

X_train = np.hstack([
    X_fp_train,
    X_desc_train
]).astype(np.float32)

X_test = np.hstack([
    X_fp_test,
    X_desc_test
]).astype(np.float32)

print(f"\nFinal feature space: {X_train.shape[1]} features")
print(f"Total usable molecules: {len(X_train) + len(X_test)}")
print("\nTrain/Test Split")
print("Train size:", len(X_train))
print("Test size:", len(X_test))

print("\nTraining class distribution")
print(pd.Series(
    label_encoder.inverse_transform(y_train)
).value_counts())



print("\nTest class distribution")
print(pd.Series(
    label_encoder.inverse_transform(y_test)
).value_counts())

# ================================================================
# CROSS-VALIDATION STRATEGY
# ================================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

# ================================================================
# MODEL DEFINITIONS
# ================================================================

models = {

    "Random Forest": RandomForestClassifier(
        random_state=SEED,
        n_jobs=1
    ),

    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        use_label_encoder=False,
        random_state=SEED,
        eval_metric="mlogloss",
        n_jobs=1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        random_state=SEED,
        n_jobs=1
    )
}

# ================================================================
# HYPERPARAMETER SEARCH SPACES
# ================================================================

param_grids = {

    "Random Forest": {
        "classifier__n_estimators": [200, 300],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2]
    },

    "XGBoost": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1],
        "classifier__subsample": [0.8, 1.0]
    },

    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10]
    }
}

# ================================================================
# STORAGE
# ================================================================

results = []
best_models = {}

os.makedirs("models", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ================================================================
# TRAINING LOOP
# ================================================================

for model_name, classifier in models.items():

    print("\n" + "=" * 60)
    print(f"MODEL: {model_name}")
    print("=" * 60)

    # ============================================================
    # PIPELINE WITH SMOTE
    # ============================================================

    pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=SEED)),
        ("classifier", classifier)
    ])

    # ============================================================
    # GRID SEARCH
    # ============================================================

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    print("\nRunning hyperparameter optimization...")

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    best_models[model_name] = best_model

    print("\nBest parameters:")
    print(grid.best_params_)

    print("\nBest CV score:")
    print(grid.best_score_)

    # ============================================================
    # CROSS-VALIDATION METRICS
    # ============================================================

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "roc_auc_ovr": "roc_auc_ovr",
    }

    cv_scores = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    print("\nCross-validation results")

    for metric in scoring.keys():
        mean_score = np.mean(cv_scores[f"test_{metric}"])
        std_score = np.std(cv_scores[f"test_{metric}"])

        print(f"{metric}: {mean_score:.3f} ± {std_score:.3f}")

    

    # ============================================================
    # TEST SET EVALUATION
    # ============================================================

    y_pred = best_model.predict(X_test)

    y_prob = best_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)

    f1 = f1_score(
       y_test,
       y_pred,
       average="macro"
  )

    roc_auc_ovr = roc_auc_score(
          y_test,
         y_prob,
         multi_class="ovr",
         average="macro"
    )

    classes = np.unique(y_train)

    y_test_bin = label_binarize(
        y_test,
        classes=classes
    )

    pr_auc = average_precision_score(
       y_test_bin,
       y_prob,
       average="macro"
    )

    print("\nTest set evaluation")

    print(f"Accuracy: {acc:.3f}")
    print(f"Macro F1: {f1:.3f}")
    print(f"ROC-AUC (OvR): {roc_auc_ovr:.3f}")
    print(f"PR-AUC: {pr_auc:.3f}")

    print("\nClassification report")
    print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
    )

    print("\nConfusion matrix")
    cm = confusion_matrix(y_test, y_pred)

    cm_df = pd.DataFrame(
        cm,
        index=label_encoder.classes_,
        columns=label_encoder.classes_
    )

    print(cm_df)
    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================

    if model_name == "Random Forest":

        rf_model = best_model.named_steps["classifier"]

        importances = rf_model.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": np.arange(len(importances)),
            "Importance": importances
        })

        importance_df = importance_df.sort_values(
            by="Importance",
            ascending=False
        )

        print("\nTop 15 important features")
        print(importance_df.head(15))

        importance_df.head(25).to_csv(
            "figures/top_rf_features.csv",
            index=False
        )

    # ============================================================
    # SAVE MODEL
    # ============================================================

    save_path = f"models/{model_name.replace(' ', '_').lower()}_pipeline.pkl"

    joblib.dump(best_model, save_path)

    print(f"\nSaved model: {save_path}")

    # ============================================================
    # STORE RESULTS
    # ============================================================

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Macro_F1": f1,
        "ROC_AUC_OvR": roc_auc_ovr,
        "PR_AUC": pr_auc
    })

# ================================================================
# FINAL MODEL COMPARISON
# ================================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print(results_df)

results_df.to_csv("figures/model_comparison.csv", index=False)

# ================================================================
# BEST MODEL SELECTION
# ================================================================

best_row = results_df.sort_values(
    by="ROC_AUC_OvR",
    ascending=False
).iloc[0]

print("\nBest overall model")
print(best_row)

print("\nPipeline completed successfully.")

# ================================================================
# END OF SCRIPT
# ================================================================

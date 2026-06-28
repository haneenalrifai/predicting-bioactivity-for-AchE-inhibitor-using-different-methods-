# methodological refinement using stratified cross-validation
#====================================================================
# first, apply stratified cross-validation to the Morgan fingerprints model only
#====================================================================
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

import joblib
import os
import random

# =============================
# 1) GLOBAL REPRODUCIBILITY
# =============================

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

# =============================
# 2) LOAD CLEANED DATA
# =============================

df = pd.read_csv(r"C:\Users\QBS-OM\OneDrive\Desktop\Videos\BIS Lectures\AIML\AIML assignment\AChE_IC50_clean_labeled.csv")

required_cols = ["canonical_smiles", "activity_class"]
df = df[required_cols].dropna().reset_index(drop=True)

print("\nDataset summary")
print(df["activity_class"].value_counts())

X_smiles = df["canonical_smiles"]
y = df["activity_class"]

# =============================
# 3) MORGAN FP GENERATION
# =============================

FP_SIZE = 2048
morgan_gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((FP_SIZE,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X_fp = X_smiles.apply(smiles_to_fp)

mask = X_fp.notnull()
X = np.stack(X_fp[mask])
y = y.loc[mask].reset_index(drop=True)

print(f"\nFinal usable samples: {len(X)}")

# =============================
# 4) TRAIN / TEST SPLIT REPORT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("\nTrain/Test split (stratified, 80/20)")
print("Train size:", len(X_train))
print("Test size:", len(X_test))
print("\nClass distribution (train)")
print(y_train.value_counts())
print("\nClass distribution (test)")
print(y_test.value_counts())

# =============================
# 5) DEFINE RF BASELINE MODEL
# =============================

rf_params = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

print("\nRandom Forest hyperparameters:")
for k,v in rf_params.items():
    print(f"{k}: {v}")

model = RandomForestClassifier(**rf_params)

# =============================
# 6) CROSS-VALIDATION (STRATIFIED)
# =============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_acc = []
cv_f1 = []

for fold,(train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    cv_acc.append(accuracy_score(y_val, preds))
    cv_f1.append(f1_score(y_val, preds, average="macro"))

print("\nCross-validation performance (5-fold)")
print(f"Accuracy: mean={np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}")
print(f"Macro-F1: mean={np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")

# =============================
# 7) FINAL TRAIN ON FULL TRAIN SET
# =============================

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Test set evaluation ===")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# =============================
# 8) SAVE MODEL + FP CONFIG
# =============================

os.makedirs("models", exist_ok=True)

joblib.dump(
    dict(
        model=model,
        fp_size=FP_SIZE,
        radius=2,
        seed=SEED
    ),
    "models/rf_fp_only_reproducible.pkl"
)

print("\nModel saved: models/rf_fp_only_reproducible.pkl")
#===================================================================
# end of script
#===================================================================
#second , apply stratified cross-validation to the combined Morgan fingerprints + descriptors model
#===================================================================
import numpy as np
import pandas as pd
import random
import joblib
import os

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# =============================
# REPRODUCIBILITY
# =============================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# =============================
# LOAD DATA
# =============================

df = pd.read_csv(r"C:\Users\QBS-OM\OneDrive\Desktop\Videos\BIS Lectures\AIML\AIML assignment\AChE_IC50_clean_labeled_more_descriptors.csv")

required_cols = [
    "canonical_smiles",
    "activity_class",
    "Molecular Weight",
    "AlogP",
    "#RO5 Violations"
]

df = df[required_cols].dropna().reset_index(drop=True)

print("\nDataset summary")
print(df["activity_class"].value_counts())

X_smiles = df["canonical_smiles"]
y = df["activity_class"]

# =============================
# MORGAN FP
# =============================

FP_SIZE = 2048
morgan_gen = GetMorganGenerator(radius=2, fpSize=FP_SIZE)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = morgan_gen.GetFingerprint(mol)
    arr = np.zeros((FP_SIZE,), dtype=int)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

X_fp = X_smiles.apply(smiles_to_fp)

mask = X_fp.notnull()
X_fp = np.stack(X_fp[mask])
y = y.loc[mask].reset_index(drop=True)

# =============================
# DESCRIPTORS (STANDARDIZED)
# =============================

desc_cols = ["Molecular Weight","AlogP","#RO5 Violations"]

scaler = StandardScaler()
X_desc = scaler.fit_transform(df.loc[mask, desc_cols])

# Combine feature spaces
X = np.hstack([X_fp, X_desc])

print(f"\nFinal feature space = {X.shape[1]} features")

# =============================
# TRAIN / TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=SEED
)

print("\nTrain/Test split (80/20)")
print("Train size:", len(X_train))
print("Test size:", len(X_test))

print("\nClass distribution (train)")
print(y_train.value_counts())

print("\nClass distribution (test)")
print(y_test.value_counts())

# =============================
# RANDOM FOREST CONFIG
# =============================

rf_params = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1
)

print("\nRandom Forest hyperparameters:")
for k,v in rf_params.items():
    print(f"{k}: {v}")

model = RandomForestClassifier(**rf_params)

# =============================
# STRATIFIED K-FOLD CV
# =============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

cv_acc = []
cv_f1 = []

for fold,(tr,vl) in enumerate(cv.split(X_train, y_train),1):
    X_tr, X_val = X_train[tr], X_train[vl]
    y_tr, y_val = y_train.iloc[tr], y_train.iloc[vl]

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)

    cv_acc.append(accuracy_score(y_val,preds))
    cv_f1.append(f1_score(y_val,preds,average="macro"))

print("\nCross-validation performance (5-fold)")
print(f"Accuracy: mean={np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}")
print(f"Macro-F1: mean={np.mean(cv_f1):.3f} ± {np.std(cv_f1):.3f}")

# =============================
# FINAL TEST EVALUATION
# =============================

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n=== Test set evaluation ===")
print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# =============================
# SAVE MODEL + SCALER + CONFIG
# =============================

os.makedirs("models", exist_ok=True)

joblib.dump(
    dict(
        model=model,
        scaler=scaler,
        desc_cols=desc_cols,
        fp_size=FP_SIZE,
        radius=2,
        seed=SEED
    ),
    "models/rf_fp_desc_reproducible.pkl"
)

print("\nModel saved: models/rf_fp_desc_reproducible.pkl")
# ===========================================================================
# end of script 
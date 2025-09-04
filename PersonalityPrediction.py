# =========================================================
# 1. Imports
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report

import xgboost as xgb

# For Kaggle output
import zipfile

# =========================================================
# 2. Load Data
# =========================================================
train = pd.read_csv("/kaggle/input/your-dataset-folder/train.csv")
test  = pd.read_csv("/kaggle/input/your-dataset-folder/test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()

# =========================================================
# 3. Identify Columns
# =========================================================
target = "Personality"
id_col = "id"

num_cols = train.select_dtypes(include=np.number).columns.tolist()
cat_cols = train.select_dtypes(include="object").columns.tolist()
cat_cols.remove(target)
cat_cols.remove(id_col)

print("Numerical cols:", num_cols)
print("Categorical cols:", cat_cols)

# =========================================================
# 4. EDA
# =========================================================
# Target distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train, x=target, palette="coolwarm")
plt.title("Target Distribution: Extrovert vs Introvert")
plt.show()

# Correlation heatmap
clean_num_cols = [col for col in num_cols if train[col].nunique() > 1 and train[col].notna().sum() > 0]
corr_matrix = train[clean_num_cols].corr().dropna(axis=0, how="all").dropna(axis=1, how="all")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# Pairplots (sampled for speed)
sns.pairplot(train.sample(500), vars=clean_num_cols[:4], hue=target, diag_kind="kde", palette="coolwarm")
plt.suptitle("Pairplots: Numerical Features vs Personality", y=1.02)
plt.show()

# =========================================================
# 5. Preprocessing
# =========================================================
X = train.drop(columns=[target, id_col])
y = train[target]
X_test = test.drop(columns=[id_col])

# Handle categorical
for col in cat_cols:
    X[col] = X[col].fillna("Missing")
    X_test[col] = X_test[col].fillna("Missing")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    X_test[col] = le.transform(X_test[col])

# Handle numerical
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
    X_test[col] = X_test[col].fillna(X[col].median())

scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Encode target
y = LabelEncoder().fit_transform(y)

# =========================================================
# 6. Train/Validation Split
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# =========================================================
# 7. Models
# =========================================================
rf = RandomForestClassifier(n_estimators=200, random_state=42)
xgb_model = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42
)

voting_model = VotingClassifier(
    estimators=[("rf", rf), ("xgb", xgb_model)],
    voting="soft"
)

voting_model.fit(X_train, y_train)

# Validation report
y_val_pred = voting_model.predict(X_val)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))

# =========================================================
# 8. Predictions on Test
# =========================================================
y_test_pred = voting_model.predict(X_test)
y_test_pred = LabelEncoder().fit(train[target]).inverse_transform(y_test_pred)

# =========================================================
# 9. Create Submission
# =========================================================
submission = pd.DataFrame({
    "id": test[id_col],
    "Personality": y_test_pred
})

# Save CSV
submission.to_csv("/kaggle/working/Predict_the_Introverts_from_the_Extroverts.csv", index=False)

# Optional: also zip it
with zipfile.ZipFile("/kaggle/working/Predict_the_Introverts_from_the_Extroverts.zip", 'w') as z:
    z.write("/kaggle/working/Predict_the_Introverts_from_the_Extroverts.csv",
            arcname="Predict_the_Introverts_from_the_Extroverts.csv")

print("Submission files created: CSV and ZIP")
submission.head()

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/extracted_features/features.csv')

X = df.drop(columns=['label', 'source']).values
le = LabelEncoder()
y = le.fit_transform(df['label'])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42))
])
model.fit(X_tr, y_tr)

pred = model.predict(X_te)
print(classification_report(y_te, pred, target_names=le.classes_))
print("Confusion matrix (rows=true, cols=predicted):")
print(f"labels order: {list(le.classes_)}")
print(confusion_matrix(y_te, pred))

names = df.drop(columns=['label', 'source']).columns
imps = model.named_steps['rf'].feature_importances_
print("\nFeature importances:")
for n, i in sorted(zip(names, imps), key=lambda x: -x[1]):
    print(f"  {n:32s} {i:.4f}")

os.makedirs('data/models', exist_ok=True)
pickle.dump({'model': model, 'encoder': le, 'feature_names': list(names)},
            open('data/models/aquaguard_rf.pkl', 'wb'))
print("\nSaved data/models/aquaguard_rf.pkl")

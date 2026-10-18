"""
Trains the Random Forest behavior classifier and reports honest metrics.

Why grouped cross-validation: training windows overlap ~83% (90-frame windows,
15-frame stride), so a random train/test split puts near-identical windows from
the same clip on both sides and inflates scores. Here every window from a given
video is either all-train or all-test, so the score reflects performance on
footage the model has never seen.

Outputs:
  data/models/aquaguard_rf.pkl      model bundle served by the API
  reports/metrics.json              grouped-CV metrics
  reports/confusion_matrix.png      for the README
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

sys.path.append('.')
from src import config
from src.features.feature_engineer import FEATURE_NAMES

FEATURES_CSV = 'data/extracted_features/features.csv'
REPORT_DIR = 'reports'


def build_model():
    return Pipeline([
        ('impute', SimpleImputer(strategy='median')),   # leg_activity is NaN when ankles are submerged
        ('rf', RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                      class_weight='balanced', random_state=42, n_jobs=-1)),
    ])


def grouped_cv(X, y, groups, classes):
    """Predict every window using a model that never saw its video."""
    vids_per_class = pd.Series(groups).groupby(y).nunique()
    n_splits = int(min(5, vids_per_class.min()))
    if n_splits < 2:
        print("\nWARNING: at least one class comes from a single video, so there is no "
              "honest held-out test for it. Add more clips per class.")
        print(vids_per_class.rename(lambda i: classes[i]).to_string())
        return None, n_splits
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_predict(build_model(), X, y, groups=groups, cv=cv), n_splits


def save_confusion_png(cm, classes, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.0%}", ha='center', va='center',
                    color='white' if norm[i, j] > 0.5 else 'black', fontsize=9)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Grouped CV (held-out videos)', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(FEATURES_CSV)
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        sys.exit(f"features.csv is missing {missing}. Re-run extract_dataset.py.")

    X = df[FEATURE_NAMES].values
    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    groups = df['video'].values
    classes = list(le.classes_)

    print(f"{len(df)} windows, {df['video'].nunique()} videos")
    print(df.groupby('label').agg(windows=('video', 'size'), videos=('video', 'nunique')), "\n")

    pred, n_splits = grouped_cv(X, y, groups, classes)
    os.makedirs(REPORT_DIR, exist_ok=True)
    metrics = {'n_windows': int(len(df)), 'n_videos': int(df['video'].nunique()),
               'classes': classes, 'window_frames': config.WINDOW, 'cv_folds': n_splits}

    if pred is not None:
        report = classification_report(y, pred, target_names=classes, output_dict=True, zero_division=0)
        cm = confusion_matrix(y, pred)
        print(f"Grouped {n_splits}-fold CV (each video held out once):")
        print(classification_report(y, pred, target_names=classes, zero_division=0))
        print(f"Confusion matrix, rows=true cols=pred, order {classes}:\n{cm}")
        # What matters operationally: is a swimmer in danger at all?
        # (drowning vs distress both mean "go"; confusing them still triggers an alert)
        normal_idx = classes.index('normal')
        true_d, pred_d = y != normal_idx, pred != normal_idx
        danger = {'recall': float((true_d & pred_d).sum() / max(true_d.sum(), 1)),
                  'precision': float((true_d & pred_d).sum() / max(pred_d.sum(), 1)),
                  'false_alarm_rate': float((~true_d & pred_d).sum() / max((~true_d).sum(), 1))}
        print(f"\nDanger vs normal: recall {danger['recall']:.1%}, precision {danger['precision']:.1%}, "
              f"false-alarm rate on normal windows {danger['false_alarm_rate']:.1%}")
        metrics.update({'report': report, 'confusion_matrix': cm.tolist(), 'danger_vs_normal': danger})
        save_confusion_png(cm, classes, f'{REPORT_DIR}/confusion_matrix.png')

    # Final model: fit on everything for serving
    model = build_model().fit(X, y)
    imps = sorted(zip(FEATURE_NAMES, model.named_steps['rf'].feature_importances_),
                  key=lambda x: -x[1])
    print("\nFeature importances:")
    for n, i in imps:
        print(f"  {n:28s} {i:.4f}")
    metrics['feature_importances'] = {n: round(float(i), 4) for n, i in imps}

    with open(f'{REPORT_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump({'model': model, 'encoder': le, 'feature_names': FEATURE_NAMES,
                     'metrics': {k: metrics[k] for k in ('n_windows', 'n_videos', 'cv_folds')}}, f)
    print(f"\nSaved {config.MODEL_PATH} and {REPORT_DIR}/metrics.json")


if __name__ == "__main__":
    main()

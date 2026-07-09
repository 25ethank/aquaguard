import pandas as pd

df = pd.read_csv('data/extracted_features/features.csv')

print("Samples per class:")
print(df['label'].value_counts())
print()

key_features = ['body_angle_mean', 'body_vertical_ratio', 'arm_symmetry', 'wrist_above_shoulder_ratio', 'leg_activity', 'head_y_trend']

print("Feature means by class:")
print(df.groupby('label')[key_features].mean().round(3))

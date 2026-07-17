import pandas as pd

# Diagnose missing distress label
# pd.set_option('display.max_colwidth', None)  # don't truncate long strings
# pd.set_option('display.max_rows', None)       # don't truncate rows

df = pd.read_csv('data/extracted_features/features.csv')

# Just look at drowning-labeled rows — which video files are actually in there?
drowning_rows = df[df['label'] == 'drowning']
print("Unique video files inside the 'drowning' label:")
print(drowning_rows['source'].str.extract(r'([^/]+\.mp4)')[0].value_counts())

print()
print("Sample of actual source strings for drowning rows:")
print(drowning_rows['source'].head(10).tolist())

print("Unique video files inside the 'normal' label:")
print(df[df['label'] == 'normal']['source'].str.extract(r'([^/]+\.mp4)')[0].value_counts())

print("\nAll unique labels currently in the CSV:")
print(df['label'].unique())

# Diagnose head_y_trend
print(df.groupby('label')['head_y_trend'].describe())

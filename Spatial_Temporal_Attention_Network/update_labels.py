import pandas as pd
from pathlib import Path

input_csv = "avec2014/labels.csv"
output_csv = "avec2014/labels_updated.csv"

df = pd.read_csv(input_csv)
df['filename'] = df['filename'].apply(lambda x: f"{Path(x).parent.name}_{Path(x).parent.parent.name}/{Path(x).stem}_aligned")
df.to_csv(output_csv, index=False)

print("Updated labels.csv!")
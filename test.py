import pandas as pd

df = pd.read_parquet('data/final/codea1_filtered.parquet')

print('Shape:', df.shape)
print()
print('Columns:', df.columns.tolist())
print()
print('First 2 rows:')

for i in range(min(2, len(df))):
    print(f'--- Row {i} ---')
    row = df.iloc[i]
    for col in df.columns:
        val = row[col]
        text = str(val)
        if len(text) > 300:
            text = text[:300] + '...'
        print(f'  {col}: {text}')
    print()
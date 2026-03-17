import pandas as pd
import glob
import os

# CSV 
csv_files = glob.glob("result*.csv")

rows = []

for file in csv_files:
    df = pd.read_csv(file, sep=',', quotechar='"', skipinitialspace=True)

    protocol_name = os.path.splitext(os.path.basename(file))[0]
    
    # left shift columns by 1
    df = df.shift(periods=1, axis=1)
    
    row = {
        'protocol': protocol_name,
        'simd': df['simd'].iloc[0],
        'avg_pre_time': df['pre-processing-time'].mean(),
        'avg_online_time': df['online-time'].mean(),
        'pre_all': df['pre-processing-all'].iloc[0],
        'online_all': df['online-all'].iloc[0]
    }
    rows.append(row)

summary = pd.DataFrame(rows)

print(summary)

summary.to_csv("summary.csv", index=False)


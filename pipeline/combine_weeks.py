"""
combine_weeks.py

Concatenates the 17 per-week output files from pipeline.py into
combined_def.csv and combined_all22.csv.
"""
import pandas as pd
import glob

def_files = sorted(glob.glob('week*_def.csv'))
all22_files = sorted(glob.glob('week*_all22.csv'))

def_all = pd.concat([pd.read_csv(f) for f in def_files], ignore_index=True)
all22_all = pd.concat([pd.read_csv(f) for f in all22_files], ignore_index=True)

print('total defender-cloud plays:', len(def_all))
print('total all22-cloud plays:', len(all22_all))
print(def_all['outcome'].value_counts())

def_all.to_csv('combined_def.csv', index=False)
all22_all.to_csv('combined_all22.csv', index=False)

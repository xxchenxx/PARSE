from glob import glob
import pandas as pd

filelist = sorted(glob("*_predicted.csv"))

finals = []

for file in filelist:
    df = pd.read_csv(file)
    df['filename'] = '_'.join(file.split("_")[:-1])
    finals.append(df)

finals = pd.concat(finals)
finals.to_csv("finals.csv", index=False)

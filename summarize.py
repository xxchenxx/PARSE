import argparse
import os

parser = argparse.ArgumentParser(description='Summarize a Python file')
parser.add_argument('--dir', type=str)

from glob import glob

args = parser.parse_args()

files = sorted(glob(os.path.join(args.dir, "ours_rnk*")))
labels = [
'1hzz',
'1hyo',
'1hzz',
'1hyo',
'1oth',
'1fcb',
'1hyo',
'1fcb',
'1fcb',
'1oth',
'1fcb',
'1oth',
'1oth']
funcs = None
import pandas as pd

f1 = open("results_mean.txt", "w")
f2 = open("results_lb.txt", "w")
for file, label in zip(files, labels):
    mean_scores = {}
    lb_scores = {}
    print(file)
    data = pd.read_csv(file)
    if funcs is None:
        funcs = data['site'].apply(lambda x: x[:4]).unique()
    for key in funcs:
        scores = data[data['site'].apply(lambda x: x[:4]) == key]['score']
        mean_score = scores.values.mean()
        lb_score = scores.values.mean() - 1.96 * scores.values.std() / len(scores.values) ** 0.5
        mean_scores[key] = mean_score
        lb_scores[key] = lb_score
    mean_results = []
    lb_results = []
    mean_results.append((label, mean_scores[label]))
    lb_results.append((label, lb_scores[label]))
    mean_scores = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
    lb_scores = sorted(lb_scores.items(), key=lambda x: x[1], reverse=True)
    
    mean_results.extend(mean_scores[:2])
    lb_results.extend(lb_scores[:2])
    
    mean_results = list(set(mean_results))
    lb_results = list(set(lb_results))
    if len(mean_results) != 3:
        mean_results = sorted(mean_results, key=lambda x: x[1], reverse=True)
    else:
        mean_results = mean_results[0:1] + sorted(mean_results[1:], key=lambda x: x[1], reverse=True)
    f1.write(f"{mean_results[1][0]}\t{mean_results[0][0]}\t{mean_results[1][1]}\t{mean_results[0][1]}\n")

    if len(lb_results) != 3:
        lb_results = sorted(lb_results, key=lambda x: x[1], reverse=True)
    else:
        lb_results = lb_results[0:1] + sorted(lb_results[1:], key=lambda x: x[1], reverse=True)
    f2.write(f"{lb_results[1][0]}\t{lb_results[0][0]}\t{lb_results[1][1]}\t{lb_results[0][1]}\n")

f1.close()
f2.close()
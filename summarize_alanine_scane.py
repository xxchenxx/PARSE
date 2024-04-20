import argparse
import os

parser = argparse.ArgumentParser(description='Summarize a Python file')
parser.add_argument('--file', type=str)
args = parser.parse_args()
file = open(args.file, "r").readlines()
result = list(map(float, [x.strip().split()[1] for x in file]))
print(f"Mean: {sum(result) / len(result)}")
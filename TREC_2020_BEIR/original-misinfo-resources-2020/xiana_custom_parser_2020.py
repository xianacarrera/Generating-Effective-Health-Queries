import argparse
from typing import Final
from xml.etree import cElementTree as ET

import pandas as pd
import os
from numpy import int64

current_directory = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--qrelsdir', default="qrels/2020-derived-qrels")

args = parser.parse_known_args()

qrels_dir = args[0].qrelsdir

directory = os.path.join(current_directory, qrels_dir)

for filename in os.listdir(directory):
    if filename.endswith(".tsv") or any(chr.isdigit() for chr in filename):
        continue
    
    f = os.path.join(current_directory, qrels_dir, filename)
    qrels = pd.read_csv(f, sep=' ',
                        names=['topic', 'iter', 'docno', 'Grade'])

    qrels["topic docno Grade".split()] \
        .to_csv(f"{os.path.join(current_directory, f)}.tsv", index=False, header=["topic", "docno", "Grade"], sep="\t")



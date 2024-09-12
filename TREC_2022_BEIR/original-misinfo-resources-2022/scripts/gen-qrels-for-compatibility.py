import argparse
from typing import Final
from xml.etree import cElementTree as ET

import pandas as pd

# %%
VERY_USEFUL: Final[int] = 2
USEFUL: Final[int] = 1
NOT_USEFUL: Final[int] = 0

JUDGED_YES: Final[int] = 1
JUDGED_NO: Final[int] = 0

JUDGED_UNCLEAR: Final[int] = 2

JUDGED_CORRECT: Final[int] = 1
JUDGED_INCORRECT: Final[int] = 0

UNJUDGED: Final[int] = -1

# %% read prefs
parser = argparse.ArgumentParser()

parser.add_argument('--qrels', default="qrels/qrels.final.oct-19-2022")
parser.add_argument('--topics', default="topics/misinfo-2022-topics.xml")
parser.add_argument('--prefs', default="qrels/trec2022_act26_v2.csv")
parser.add_argument('--output', default="qrels/2022-derived-qrels/misinfo-qrels")

args = parser.parse_known_args()

qrels_file = args[0].qrels
topics_file = args[0].topics
prefs_file = args[0].prefs
output_file = args[0].output

# %%
# https://stackoverflow.com/a/10077069/5935439
from collections import defaultdict

from numpy import int64


def etree_to_dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = defaultdict(list)
        for dc in map(etree_to_dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v) for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


with open(topics_file) as f:
    topics_dict = etree_to_dict(ET.XML(f.read()))["topics"]['topic']
topics = pd.DataFrame(topics_dict)
topics = topics.rename(columns={"number": "topic"})
topics.topic = topics.topic.astype(int64)

df_prefs = pd.read_csv(prefs_file)
topics["Topic ID"] = topics.apply(lambda x: f"{x.question} (Answer is {x.answer.capitalize()})", axis=1)
df_prefs = df_prefs.merge(
    topics[["topic", "Topic ID", "answer"]],
    on="Topic ID",
    how="inner"
)
df_prefs = df_prefs.rename(columns={"Document UUID": "docno"})

# %% read qrels
qrels = pd.read_csv(qrels_file, sep=' ',
                    names=['topic', 'docno', 'usefulness', 'answer_judged'])
qrels = qrels.merge(topics["topic answer".split()], on="topic", how="inner")
correctness = {
    (JUDGED_YES, "yes"): JUDGED_CORRECT,
    (JUDGED_NO, "no"): JUDGED_CORRECT,
    (JUDGED_YES, "no"): JUDGED_INCORRECT,
    (JUDGED_NO, "yes"): JUDGED_INCORRECT,
    (JUDGED_UNCLEAR, "yes"): JUDGED_UNCLEAR,
    (JUDGED_UNCLEAR, "no"): JUDGED_UNCLEAR,
}

qrels["correctness"] = qrels.apply(
    lambda x: correctness.get((x.answer_judged, x.answer), UNJUDGED),
    axis=1)

grades = {
    (VERY_USEFUL, JUDGED_CORRECT): 4,
    (USEFUL, JUDGED_CORRECT): 3,
    (VERY_USEFUL, JUDGED_UNCLEAR): 2,
    (USEFUL, JUDGED_UNCLEAR): 1,
    (VERY_USEFUL, UNJUDGED): 2,  # there are no cases of this
    (USEFUL, UNJUDGED): 1,  # there is one case of this for some reason
    (NOT_USEFUL, UNJUDGED): 0,
    (USEFUL, JUDGED_INCORRECT): -1,
    (VERY_USEFUL, JUDGED_INCORRECT): -2,
}

qrels["qrel_grade"] = qrels.apply(lambda x: grades[(x.usefulness, x.correctness)], axis=1)

qrels = qrels.merge(
    df_prefs["topic docno Grade".split()],
    on="topic docno".split(),
    how="left"
)

qrels.Grade = qrels.groupby("topic").Grade.transform("max").sub(qrels.Grade).add(max(grades.values()) + 1)

qrels.Grade = qrels.Grade.fillna(qrels["qrel_grade"]).astype("int")

qrels.insert(1, "iter", 0)

qrels = qrels.sort_values("topic Grade".split(), ascending=[True, False])
qrels["topic iter docno Grade".split()] \
    .to_csv(f"{output_file}.graded", index=False, header=False, sep=" ")

temp = qrels[qrels.Grade < 0]["topic iter docno Grade".split()]
temp.Grade = temp.Grade * -1
temp.to_csv(f"{output_file}.graded-harmful-only", index=False, header=False, sep=" ")


temp = qrels[qrels.Grade > 0]["topic iter docno Grade".split()]
temp.to_csv(f"{output_file}.graded-helpful-only", index=False, header=False, sep=" ")

temp = qrels[qrels.correctness == JUDGED_CORRECT]["topic iter docno".split()]
temp.insert(3, "correct", 1)
temp.to_csv(f"{output_file}.binary-useful-correct", index=False, header=False, sep=" ")

temp = qrels[qrels.correctness == JUDGED_INCORRECT]["topic iter docno".split()]
temp.insert(3, "incorrect", 1)
temp.to_csv(f"{output_file}.binary-incorrect", index=False, header=False, sep=" ")

temp = qrels[qrels.usefulness >= 0]["topic iter docno usefulness".split()]
temp.to_csv(f"{output_file}.graded-usefulness", index=False, header=False, sep=" ")

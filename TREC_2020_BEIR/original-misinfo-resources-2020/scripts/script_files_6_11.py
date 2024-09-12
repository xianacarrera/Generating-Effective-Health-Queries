# Required Packages `pandas`
# Install using `pip install pandas`

# If using computecanada you can use the command `module load scipy-stack`
# to load an interpreter with pandas preinstalled

# Example Usage:
# python .\script_files_6_10.py --qrels .\misinfo-qrels.judged --topics .\topics.xml --output output


import argparse

import sys

if sys.version_info[0] < 3:
    print("""Python < 3 is unsupported.
If using compute canada type `module load scipy-stack`
to switch interpreters""")
    sys.exit(-1)

from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("""Package pandas not installed.
If using compute canada type `module load scipy-stack`
to switch interpreters
otherwise install using `pip install pandas`""")
    sys.exit(-1)

import xml.etree.ElementTree as et

parser = argparse.ArgumentParser(description='Parse Health misinformation track qrels')
parser.add_argument('--qrels', type=Path, help='Path to qrels file', required=True)
parser.add_argument('--topics', type=Path, help='Path to topics file', required=True)
# parser.add_argument('--qrels', type=Path, help='Path to qrels file', default='misinfo-qrels.fixed')
# parser.add_argument('--topics', type=Path, help='Path to topics file', default='topics.xml')
parser.add_argument('--output', type=Path, help='Output directory', default='output')

args = parser.parse_known_args()[0]

try:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(e)
    sys.exit(-1)

try:
    # Read Qrels
    _qrels = pd.read_csv(args.qrels, sep=' ',
                         names=['topic.id', 'iteration', 'docno', 'usefulness', 'doc.answer', 'credibility'])

    # Read topics
    root = et.parse(args.topics).getroot()
except OSError as e:
    print(e)
    sys.exit(-1)

topic2answer = dict(zip(
    [int(x.text) for x in root.findall('.//number')],
    [x.text for x in root.findall('.//answer')]
))

# File 1
usefulness = _qrels.usefulness.map({0: 'not.useful', 1: 'useful'})
doc_answer = a = _qrels["doc.answer"].map({0: 'none', 1: 'yes', 2: 'no', -1: 'NA'})
answer = _qrels["topic.id"].map(topic2answer)
correct = (answer == doc_answer).map({True: 'TRUE', False: 'FALSE'}).mask(doc_answer.isin(['none', 'NA']), 'NA')
credibility = _qrels.credibility.map({0: 'not.credible', 1: 'credible', -1: 'NA'})
qrels = pd.concat(
    [_qrels["topic.id"], answer, _qrels.docno, usefulness, doc_answer, correct, credibility],
    axis=1,
    keys=["topic.id", "answer", "docno", "usefulness", "doc.answer", "correct", "credibility"]
)

assert not qrels.isnull().any().any()
assert not (usefulness.eq('not.useful') & correct.eq('TRUE')).any()
assert not (usefulness.eq('not.useful') & (doc_answer.eq('yes') | doc_answer.eq('no') | doc_answer.eq('none'))).any()
assert not (usefulness.eq('not.useful') & (credibility.eq('not.credible') | credibility.eq('credible'))).any()

qrels.to_csv(output_dir / 'misinfo-qrels.for-R.txt', sep='\t', index=False)

# File 6
qrels_binary_useful_correct = ((usefulness == 'useful') & (correct == 'TRUE')).map({True: 1, False: 0}).rename(
    'useful.correct')
qrels_binary_useful_correct = pd.concat([_qrels.iloc[:, 0:3], qrels_binary_useful_correct], 1)

qrels_binary_useful_correct.to_csv(output_dir / "misinfo-qrels-binary.useful-correct", index=False, sep=' ',
                                   header=False)

# File 7
df = pd.concat([
    qrels["topic.id"],
    qrels.usefulness.eq('useful').rename("useful"),
    qrels.usefulness.eq('not.useful').rename('not.useful'),
    qrels["doc.answer"].eq('none').rename('answer.missing'),
    qrels["doc.answer"].eq('yes').rename('answer.yes'),
    qrels["doc.answer"].eq('no').rename('answer.no'),
    qrels["doc.answer"].eq('NA').rename('answer.unjudged'),
    qrels.correct.eq('TRUE'),
    qrels.correct.eq('FALSE').rename("not.correct"),
    qrels.credibility.eq('credible').rename('credible'),
    qrels.credibility.eq('not.credible').rename('not.credible'),
    qrels.credibility.eq('NA').rename('credibility.unjudged'),
], axis=1).astype(int).groupby(["topic.id"]).sum()

df = qrels.groupby('topic.id')['topic.id'].count().rename('total').to_frame().merge(df, left_index=True,
                                                                                    right_index=True)
df = df.join(pd.Series(topic2answer, name='answer.given'), how='inner', on='topic.id')
df.insert(7, "answer.given", df.pop("answer.given"))

assert df.total.eq(df["useful"] + df["not.useful"]).all()
assert df.total.eq(df["answer.missing"] + df["answer.yes"] + df["answer.no"] + df["answer.unjudged"]).all()
assert df.total.eq(df["correct"] + df["not.correct"] + df["answer.missing"] + df["answer.unjudged"]).all()
assert df.total.eq(df["credible"] + df["not.credible"] + df["credibility.unjudged"]).all()

df.to_csv(output_dir / 'misinfo-qrels.counts.txt', sep='\t')

# File 8
_qrels['answer'] = qrels['answer']


def is_correct(x):
    if x["doc.answer"] == 1 and x["answer"] == "yes": return 1
    if x["doc.answer"] == 2 and x["answer"] == "no": return 1
    return 0


aspect_correct = _qrels.apply(is_correct, 1).rename("correct")
aspect_credible = _qrels.iloc[:, 5].map({1: 1, 0: 0, -1: 0})

qrels_3aspects = pd.concat([_qrels.iloc[:, 0:4], aspect_correct, aspect_credible], 1)
qrels_3aspects.to_csv("misinfo-qrels.3aspects", index=False, sep=' ', header=False)

# File 9
qrels_2aspects_uc = qrels_3aspects.iloc[:, [0, 1, 2, 3, 5]]
qrels_2aspects_uc.to_csv(output_dir / "misinfo-qrels.2aspects.useful-credible", index=False, sep=' ', header=False)

# File 10
qrels_2aspects_cc = qrels_3aspects.iloc[:, [0, 1, 2, 4, 5]]
qrels_2aspects_cc.to_csv(output_dir / "misinfo-qrels.2aspects.correct-credible", index=False, sep=' ', header=False)

# File 11
_qrels['answer'] = qrels['answer']


def g(x):
    if x["usefulness"] == 0: return 0
    if x["usefulness"] == 1:
        if x["doc.answer"] == 0 or x["doc.answer"] == -1: return 0
        if x["doc.answer"] == 2 and x["answer"] == "yes": return 1
        if x["doc.answer"] == 1 and x["answer"] == "no": return 1
    return 0


aspect_incorrect = _qrels.apply(g, 1).rename("incorrect")

qrels_binary_incorrect = pd.concat([_qrels.iloc[:, 0:3], aspect_incorrect], 1)

qrels_binary_incorrect.to_csv(output_dir / "misinfo-qrels-binary.incorrect", index=False, sep=' ', header=False)

# Extra Sanity Checks
assert qrels_3aspects.credibility.astype(bool).eq(qrels.credibility.eq("credible")).all()
assert qrels_3aspects.correct.eq(qrels.correct.map({'NA': 0, 'FALSE': 0, 'TRUE': 1})).all()
assert qrels_3aspects.usefulness.astype(bool).eq(qrels.usefulness.eq('useful')).all()
assert (qrels_3aspects.usefulness == qrels_2aspects_uc.usefulness).all()
assert (qrels_3aspects.correct == qrels_2aspects_cc.correct).all()
assert (qrels_3aspects.credibility == qrels_2aspects_cc.credibility).all()
assert qrels.shape[0] == qrels_3aspects.shape[0]
assert qrels.shape[0] == _qrels.shape[0]
assert qrels_binary_useful_correct["useful.correct"].eq(qrels_3aspects.correct & qrels_3aspects.usefulness).all()
assert not (qrels_binary_incorrect.incorrect & correct.eq("TRUE")).any()
assert ((qrels_binary_incorrect.incorrect | correct.eq("TRUE")) | (doc_answer.eq('NA') | doc_answer.eq('none'))).all()

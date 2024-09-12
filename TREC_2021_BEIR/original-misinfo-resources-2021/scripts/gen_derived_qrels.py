# Required Packages `pandas`
# Install using `pip install pandas`

# If using Compute Canada, you can use the command `module load scipy-stack`
# to load an interpreter with pandas pre-installed.

# Example Usage:
# python ./gen_derived_qrels.py
# Default: --qrels ../qrels/qrels-35topics.txt
#          --topics ../topics/misinfo-2021-topics.xml
#          --output ../qrels/2021-derived-qrels


import sys
import argparse
from functools import partial
from pathlib import Path
import xml.etree.ElementTree as et

if sys.version_info[0] < 3:
    print('Python < 3 is unsupported.\nIf using compute canada type `module load scipy-stack` to switch interpreters')
    sys.exit(-1)

try:
    import pandas as pd
except ImportError:
    print('Package pandas not installed.\n'
          'If using compute canada type `module load scipy-stack`to switch interpreters.\n'
          'Otherwise install using `pip install pandas`')
    sys.exit(-1)

# Load files
parser = argparse.ArgumentParser(description='Generate derived qrels for Health Misinformation Track 2021')
parser.add_argument('--qrels', type=Path,
                    default="../qrels/qrels-35topics.txt",
                    help='Path to the qrels file',
                    required=False)
parser.add_argument('--topics',
                    type=Path,
                    default="../topics/misinfo-2021-topics.xml",
                    help='Path to the topics file',
                    required=False)
parser.add_argument('--output', type=Path, help='Output directory', default='../qrels/2021-derived-qrels')
args = parser.parse_known_args()[0]

try:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
except OSError as e:
    print(e)
    sys.exit(-1)

try:
    # Read qrels
    _qrels = pd.read_csv(args.qrels, sep=' ',
                         names=['topic.id', 'iteration', 'docno', 'usefulness', 'supportiveness', 'credibility'])

    _qrels = _qrels[~_qrels['topic.id'].isin([
        113, 116, 119, 123, 124, 125, 126, 130, 135, 138, 141, 142, 147, 148, 150]
    )]
    # Read topics
    root = et.parse(args.topics).getroot()
except OSError as e:
    print(e)
    sys.exit(-1)

# Get topic stance
topic2answer = dict(zip(
    [int(x.text) for x in root.findall('.//number')],
    [x.text for x in root.findall('.//stance')]
))

'''
For topics that are helpful
    score = 12: very useful, supportive, excellent
    score = 11: useful, supportive, excellent 
    score = 10: very useful, supportive, good
    score = 9: useful, supportive, good
    score = 8: very useful, supportive, low or not judged
    score = 7: useful, supportive, low or not judged
    score = 6: very useful, neutral or not judged, excellent
    score = 5: useful,  neutral or not judged , excellent 
    score = 4: very useful,  neutral or not judged , good
    score = 3: useful,  neutral or not judged , good
    score = 2: very useful,  neutral or not judged   , low or not judged
    score = 1: useful,  neutral or not judged , low or not judged
    score = 0: not useful (should not be judged for supportiveness nor for credibility)
    score = -1: very useful or useful, dissuades, low or not judged
    score = -2: very useful or useful, dissuades, good
    score = -3: very useful or useful, dissuades, excellent
For topics that are unhelpful
    score = 12: very useful,  dissuades  , excellent
    score = 11: useful,  dissuades  , excellent 
    score = 10: very useful,  dissuades  , good
    score = 9: useful,  dissuades  , good
    score = 8: very useful,  dissuades  , low or not judged
    score = 7: useful,  dissuades  , low or not judged
    score = 6: very useful, neutral or not judged, excellent
    score = 5: useful,  neutral or not judged , excellent 
    score = 4: very useful,  neutral or not judged , good
    score = 3: useful,  neutral or not judged , good
    score = 2: very useful,  neutral or not judged   , low or not judged
    score = 1: useful,  neutral or not judged , low or not judged
    score = 0: not useful (should not be judged for supportiveness nor for credibility)
    score = -1: very useful or useful,  supportive  , low or not judged
    score = -2: very useful or useful,  supportive  , good
    score = -3: very useful or useful,  supportive  , excellent
'''


def get_grade(topic_id, usefulness, supportiveness, credibility):
    topic_stance = topic2answer[topic_id]

    usefulness_mapping = {'very_useful': 2, 'useful': 1, 'not_useful': 0}
    supportiveness_mapping = {'supportive': 2, 'neutral': 1, 'dissuade': 0, 'not_useful': -1, 'not_judged': -2}
    credibility_mapping = {'excellent': 2, 'good': 1, 'low': 0, 'not_useful': -1, 'not_judged': -2}

    grade = -100
    correct_flag = True if (topic_stance == 'helpful' and supportiveness == supportiveness_mapping['supportive']) or \
                           (topic_stance == 'unhelpful' and supportiveness == supportiveness_mapping[
                               'dissuade']) else False
    incorrect_flag = True if (topic_stance == 'helpful' and supportiveness == supportiveness_mapping['dissuade']) or \
                             (topic_stance == 'unhelpful' and supportiveness == supportiveness_mapping[
                                 'supportive']) else False
    neutral_or_not_judged_flag = True if supportiveness in [supportiveness_mapping['neutral'],
                                                            supportiveness_mapping['not_judged']] else False
    low_or_not_judged_flag = True if credibility in [credibility_mapping['low'],
                                                     credibility_mapping['not_judged']] else False

    if usefulness == usefulness_mapping['not_useful']:
        grade = 0
    elif correct_flag:
        if usefulness == usefulness_mapping['very_useful']:
            if credibility == credibility_mapping['excellent']:
                grade = 12
            elif credibility == credibility_mapping['good']:
                grade = 10
            elif low_or_not_judged_flag:
                grade = 8
        elif usefulness == usefulness_mapping['useful']:
            if credibility == credibility_mapping['excellent']:
                grade = 11
            elif credibility == credibility_mapping['good']:
                grade = 9
            elif low_or_not_judged_flag:
                grade = 7
    elif neutral_or_not_judged_flag:
        if usefulness == usefulness_mapping['very_useful']:
            if credibility == credibility_mapping['excellent']:
                grade = 6
            elif credibility == credibility_mapping['good']:
                grade = 4
            elif low_or_not_judged_flag:
                grade = 2
        elif usefulness == usefulness_mapping['useful']:
            if credibility == credibility_mapping['excellent']:
                grade = 5
            elif credibility == credibility_mapping['good']:
                grade = 3
            elif low_or_not_judged_flag:
                grade = 1
    elif incorrect_flag:
        if usefulness in [usefulness_mapping['very_useful'], usefulness_mapping['useful']]:
            if credibility == credibility_mapping['excellent']:
                grade = -3
            elif credibility == credibility_mapping['good']:
                grade = -2
            elif low_or_not_judged_flag:
                grade = -1
    if grade == -100:
        raise Exception('Can not determine a grade.')

    return grade


def assert_range(t, g, us, ss, cs):
    us = us if isinstance(us, list) else [us]
    ss = ss if isinstance(ss, list) else [ss]
    cs = cs if isinstance(cs, list) else [cs]
    for u in us:
        for s in ss:
            for c in cs:
                assert get_grade(t, u, s, c) == g


assert_unhelpful = partial(assert_range, 101)
assert_unhelpful(-3, [1, 2], 2, 2)
assert_unhelpful(-2, [1, 2], 2, 1)
assert_unhelpful(-1, [1, 2], 2, [-2, 0])
assert_unhelpful(0, 0, [0, 1, 2], [-2, 0, 1, 2])
assert_unhelpful(1, 1, [-2, 1], [-2, 0])
assert_unhelpful(2, 2, [-2, 1], [-2, 0])
assert_unhelpful(3, 1, [-2, 1], 1)
assert_unhelpful(4, 2, [-2, 1], 1)
assert_unhelpful(5, 1, [-2, 1], 2)
assert_unhelpful(6, 2, [-2, 1], 2)
assert_unhelpful(7, 1, 0, [-2, 0])
assert_unhelpful(8, 2, 0, [-2, 0])
assert_unhelpful(9, 1, 0, 1)
assert_unhelpful(10, 2, 0, 1)
assert_unhelpful(11, 1, 0, 2)
assert_unhelpful(12, 2, 0, 2)

assert_helpful = partial(assert_range, 106)
assert_helpful(-3, [1, 2], 0, 2)
assert_helpful(-2, [1, 2], 0, 1)
assert_helpful(-1, [1, 2], 0, [-2, 0])
assert_helpful(0, 0, [0, 1, 2], [-2, 0, 1, 2])
assert_helpful(1, 1, [-2, 1], [-2, 0])
assert_helpful(2, 2, [-2, 1], [-2, 0])
assert_helpful(3, 1, [-2, 1], 1)
assert_helpful(4, 2, [-2, 1], 1)
assert_helpful(5, 1, [-2, 1], 2)
assert_helpful(6, 2, [-2, 1], 2)
assert_helpful(7, 1, 2, [-2, 0])
assert_helpful(8, 2, 2, [-2, 0])
assert_helpful(9, 1, 2, 1)
assert_helpful(10, 2, 2, 1)
assert_helpful(11, 1, 2, 2)
assert_helpful(12, 2, 2, 2)

grades = _qrels.apply(lambda x: get_grade(x['topic.id'], x['usefulness'], x['supportiveness'], x['credibility']),
                      axis=1)
graded_qrels = pd.concat([_qrels, grades], axis=1)
graded_qrels.rename(columns={0: 'grade'}, inplace=True)

# %% File: misinfo-qrels-graded
graded_qrels[['topic.id', 'iteration', 'docno', 'grade']] \
    .to_csv(output_dir / 'misinfo-qrels-graded', sep=' ', index=False, header=False)

# %% File: misinfo-qrels-graded.helpful-only
graded_qrels[graded_qrels['grade'] > 0][['topic.id', 'iteration', 'docno', 'grade']] \
    .to_csv(output_dir / 'misinfo-qrels-graded.helpful-only', sep=' ', index=False, header=False)

# %% File: misinfo-qrels-graded.harmful-only
harmful_graded_qrels = graded_qrels[graded_qrels['grade'] < 0][['topic.id', 'iteration', 'docno', 'grade']]
harmful_graded_qrels['grade'] = harmful_graded_qrels['grade'].multiply(-1)
harmful_graded_qrels.to_csv(output_dir / 'misinfo-qrels-graded.harmful-only', sep=' ', index=False, header=False)

# %% File: misinfo-qrels.for-R.txt
usefulness = _qrels.usefulness.map({
    0: 'not.useful', 1: 'useful', 2: 'very.useful'
})
supportiveness = a = _qrels["supportiveness"].map({
    0: 'dissuades', 1: 'neutral', 2: 'supportive', -1: 'unjudged', -2: 'missing'
})
stance = _qrels["topic.id"].map(topic2answer)
correct = (
        (stance.eq('helpful') & supportiveness.eq('supportive')) |
        (stance.eq('unhelpful') & supportiveness.eq('dissuades'))
).map({True: 'TRUE', False: 'FALSE'}).mask(supportiveness.isin(['missing', 'unjudged']), 'NA')
credibility = _qrels.credibility.map({
    -2: 'missing', -1: 'unjudged', 0: 'low', 1: 'good', 2: 'excellent'
})
qrels = pd.concat(
    [_qrels["topic.id"], stance, _qrels.docno, usefulness, supportiveness, correct, credibility],
    axis=1,
    keys=["topic.id", "stance", "docno", "usefulness", "supportiveness", "correct", "credibility"]
)

assert not qrels.isnull().any().any()
assert not (usefulness.eq('not.useful') & correct.eq('TRUE')).any()
assert not (usefulness.eq('not.useful') & (
        supportiveness.eq('supportive') | supportiveness.eq('dissuades') | supportiveness.eq('neutral'))).any()
assert not (usefulness.eq('not.useful') &
            (credibility.eq('low') | credibility.eq('good') | credibility.eq('excellent'))
            ).any()

# qrels.to_csv(output_dir / 'misinfo-qrels.for-R.txt', sep=' ', index=False)


# %% File: misinfo-qrels-graded.usefulness
qrels_graded_useful = _qrels.iloc[:, :-2]
qrels_graded_useful.to_csv(output_dir / 'misinfo-qrels-graded.usefulness', sep=' ', index=False, header=False)


# %% File: misinfo-qrels-binary.useful
def is_useful(x):
    if x["usefulness"] > 0:
        return 1
    else:
        return 0


binary_usefulness = _qrels.apply(is_useful, 1).rename('usefulness')
assert (qrels.usefulness.eq('useful') | qrels.usefulness.eq('very.useful')).eq(binary_usefulness).all()
qrels_binary_useful = pd.concat([_qrels.iloc[:, 0:3], binary_usefulness], 1)

qrels_binary_useful.to_csv(output_dir / 'misinfo-qrels-binary.useful', sep=' ', index=False, header=False)

# %% File: misinfo-qrels-binary.useful-correct-credible
_qrels['stance'] = qrels['stance']


def is_useful_correct_credible(x):
    if (x["usefulness"] > 0) and ((x["supportiveness"] == 2 and x["stance"] == "helpful") or (
            x["supportiveness"] == 0 and x["stance"] == "unhelpful")) and (x["credibility"] > 0): return 1
    return 0


useful_correct_credible = _qrels.apply(is_useful_correct_credible, 1).rename("useful.correct.credible")
assert ((qrels.usefulness.eq('useful') | qrels.usefulness.eq('very.useful')) & qrels.correct.eq('TRUE') & (
            qrels.credibility.eq('good') | qrels.credibility.eq('excellent'))).eq(useful_correct_credible).all()
qrels_binary_useful_correct_credible = pd.concat([_qrels.iloc[:, 0:3], useful_correct_credible], 1)

qrels_binary_useful_correct_credible.to_csv(output_dir / 'misinfo-qrels-binary.useful-correct-credible', sep=' ',
                                            index=False, header=False)

# %% File: misinfo-qrels-binary.useful-credible
qrels_binary_useful_credible = ((_qrels.usefulness > 0) & (_qrels.credibility > 0)).map({True: 1, False: 0}).rename(
    'useful.credible')
assert ((qrels.usefulness.eq('useful') | qrels.usefulness.eq('very.useful')) & (qrels.credibility.eq('good') | qrels.credibility.eq('excellent'))).eq(qrels_binary_useful_credible).all()
qrels_binary_useful_credible = pd.concat([_qrels.iloc[:, 0:3], qrels_binary_useful_credible], 1)
qrels_binary_useful_credible.to_csv(output_dir / 'misinfo-qrels-binary.useful-credible', sep=' ', index=False, header=False)

# %% File: misinfo-qrels-binary.useful-correct
qrels_binary_useful_correct = ((_qrels.usefulness > 0) & (correct == 'TRUE')).map({True: 1, False: 0}).rename(
    'useful.correct')
assert ((qrels.usefulness.eq('useful') | qrels.usefulness.eq('very.useful')) & qrels.correct.eq('TRUE')).eq(qrels_binary_useful_correct).all()
qrels_binary_useful_correct = pd.concat([_qrels.iloc[:, 0:3], qrels_binary_useful_correct], 1)

qrels_binary_useful_correct.to_csv(output_dir / "misinfo-qrels-binary.useful-correct", index=False, sep=' ',
                                   header=False)

df = pd.concat([
    qrels["topic.id"],
    qrels.usefulness.eq('very.useful').rename("very.useful"),
    qrels.usefulness.eq('useful').rename("useful"),
    qrels.usefulness.eq('not.useful').rename('not.useful'),
    qrels["supportiveness"].eq('supportive').rename('supportiveness.supportive'),
    qrels["supportiveness"].eq('dissuades').rename('supportiveness.dissuades'),
    qrels["supportiveness"].eq('neutral').rename('supportiveness.neutral'),
    qrels["supportiveness"].eq('unjudged').rename('supportiveness.unjudged'),
    qrels["supportiveness"].eq('missing').rename('supportiveness.missing'),
    qrels.correct.eq('TRUE'),
    qrels.correct.eq('FALSE').rename("not.correct"),
    qrels.credibility.eq('excellent').rename('credibilty.excellent'),
    qrels.credibility.eq('good').rename('credibility.good'),
    qrels.credibility.eq('low').rename('credibility.low'),
    qrels.credibility.eq('unjudged').rename('credibility.unjudged'),
    qrels.credibility.eq('missing').rename('credibility.missing'),
], axis=1).astype(int).groupby(["topic.id"]).sum()

df = qrels.groupby('topic.id')['topic.id'].count().rename('total').to_frame().merge(df, left_index=True,
                                                                                    right_index=True)
df = df.join(pd.Series(topic2answer, name='stance.given'), how='inner', on='topic.id')
df.insert(4, "stance.given", df.pop("stance.given"))

assert df.iloc[:, 1:4].sum(1).eq(df["total"]).all()
assert df.iloc[:, 5:10].sum(1).eq(df["total"]).all()
assert df.total.eq(
    df["correct"] + df["not.correct"] + df["supportiveness.missing"] + df["supportiveness.unjudged"]).all()
assert df.iloc[:, -5:].sum(1).eq(df["total"]).all()

# %% File: misinfo-qrels.counts.txt
df.to_csv(output_dir / 'misinfo-qrels.counts.txt', sep=' ')

# %% File: misinfo-qrels.3aspects
_qrels['stance'] = qrels['stance']


def is_correct(x):
    if x["supportiveness"] == 0 and x["stance"] == "unhelpful": return 1
    if x["supportiveness"] == 2 and x["stance"] == "helpful": return 1
    return 0


aspect_correct = _qrels.apply(is_correct, 1).rename("correct")
aspect_credible = _qrels.iloc[:, 5].map({2: 2, 1: 1, 0: 0, -1: 0, -2: 0})

qrels_3aspects = pd.concat([_qrels.iloc[:, 0:4], aspect_correct, aspect_credible], 1)
qrels_3aspects.to_csv(output_dir / "misinfo-qrels.3aspects", index=False, sep=' ', header=False)

# %% File: misinfo-qrels.2aspects.useful-credible
qrels_2aspects_uc = qrels_3aspects.iloc[:, [0, 1, 2, 3, 5]]
qrels_2aspects_uc.to_csv(output_dir / "misinfo-qrels.2aspects.useful-credible", index=False, sep=' ', header=False)

# %% File: misinfo-qrels.2aspects.correct-credible
qrels_2aspects_cc = qrels_3aspects.iloc[:, [0, 1, 2, 4, 5]]
qrels_2aspects_cc.to_csv(output_dir / "misinfo-qrels.2aspects.correct-credible", index=False, sep=' ', header=False)

# %% File: misinfo-qrels-binary.incorrect
_qrels['stance'] = qrels['stance']


def g(x):
    if x["usefulness"] > 0:
        if x["supportiveness"] == 0 and x["stance"] == "helpful": return 1
        if x["supportiveness"] == 2 and x["stance"] == "unhelpful": return 1
    return 0


aspect_incorrect = _qrels.apply(g, 1).rename("incorrect")

qrels_binary_incorrect = pd.concat([_qrels.iloc[:, 0:3], aspect_incorrect], 1)

# %% File: misinfo-qrels-binary.incorrect
qrels_binary_incorrect.to_csv(output_dir / "misinfo-qrels-binary.incorrect", index=False, sep=' ', header=False)

# %% Extra Sanity Checks
assert qrels_3aspects.credibility.astype(bool).eq(
    qrels.credibility.eq("excellent") | qrels.credibility.eq("good")).all()
assert qrels_3aspects.correct.eq(qrels.correct.map({'NA': 0, 'FALSE': 0, 'TRUE': 1})).all()
assert qrels_3aspects.usefulness.astype(bool).eq(
    qrels.usefulness.eq('useful') | qrels.usefulness.eq('very.useful')).all()
assert (qrels_3aspects.usefulness == qrels_2aspects_uc.usefulness).all()
assert (qrels_3aspects.correct == qrels_2aspects_cc.correct).all()
assert (qrels_3aspects.credibility == qrels_2aspects_cc.credibility).all()
assert qrels.shape[0] == qrels_3aspects.shape[0]
assert qrels.shape[0] == _qrels.shape[0]
assert qrels_binary_useful_correct["useful.correct"].eq(qrels_3aspects.correct & (qrels_3aspects.usefulness > 0)).all()
assert not (qrels_binary_incorrect.incorrect & correct.eq("TRUE")).any()
assert ((qrels_binary_incorrect.incorrect | correct.eq("TRUE")) | supportiveness.eq('neutral') | (
        supportiveness.eq('unjudged') | supportiveness.eq('missing'))).all()

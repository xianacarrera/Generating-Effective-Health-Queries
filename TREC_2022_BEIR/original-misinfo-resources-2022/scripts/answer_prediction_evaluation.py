import argparse
import numpy as np
import pandas as pd
import xml.etree.cElementTree as et
from sklearn.metrics import roc_curve, auc


# Arguments for the path of the topic file and the run file under evaluation.
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--topic_path', help='Path to the topic file', required=True)
parser.add_argument('-r', '--run_path', help='Path to the run file under evaluation', required=True)
args = parser.parse_args()


# Load correct answers provided by track organizers
xml_root = et.parse(args.topic_path)
rows = xml_root.findall('topic')
xml_data = [[int(row.find('number').text), row.find('query').text, row.find('question').text,
             row.find('background').text, row.find('answer').text] for row in rows]
topics = pd.DataFrame(xml_data, columns=['topic_id', 'query', 'question', 'background', 'answer'])
topics['answer_integer'] = topics['answer'].map({'yes': 1, 'no': 0})


# Load the run file
run = pd.read_csv(args.run_path, names=['topic_id', 'answer', 'yes_prob', 'run_name'], delim_whitespace=True)

y_pred = run['answer'].map({'yes': 1, 'no': 0})
y_target = topics['answer_integer']
y_prob = run['yes_prob']

false_positive = np.sum((y_pred == 1) & (y_target == 0))
true_positive = np.sum((y_pred == 1) & (y_target == 1))
false_negative = np.sum((y_pred == 0) & (y_target == 1))
true_negative = np.sum((y_pred == 0) & (y_target == 0))

assert true_positive + false_negative > 0
assert false_positive + true_negative > 0

true_positive_rate = true_positive / (true_positive + false_negative)
false_positive_rate = false_positive / (false_positive + true_negative)
acc = (true_positive + true_negative) / len(y_pred)
fpr, tpr, thresholds = roc_curve(y_target, y_prob)
auc_value = auc(fpr, tpr)

run_id = str(args.run_path).split('/')[-1]
print(f'{run_id}\tTPR\t{true_positive_rate: 0.4f}')
print(f'{run_id}\tFPR\t{false_positive_rate: 0.4f}')
print(f'{run_id}\tAccuracy\t{acc: 0.4f}')
print(f'{run_id}\tAUC\t{auc_value: 0.4f}')



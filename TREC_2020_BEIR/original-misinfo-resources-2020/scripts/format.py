"""Script to reformat TREC HEALTH MISINFORMATION TRACK Qrels file.

Takes as argument the track's qrels file,
the topics.xml file and either output file or print option.
See --help (-h) for more information on the reformatting options.
"""

from __future__ import print_function

import argparse
import csv
import xml.etree.ElementTree as ET

from pathlib import Path

OUTPUT_SEPARATOR = "\t"
TOPICS = {}


def load_topics(topics_filepath):
    global TOPICS
    root = ET.parse(topics_filepath).getroot()
    for idx, log_element in enumerate(root.findall('topic')):
        number = int(log_element.find('number').text)
        TOPICS[number] = {
            "number": number,
            "title": log_element.find('title').text,
            "description": log_element.find('description').text,
            "answer": log_element.find('answer').text,
            "evidence": log_element.find('evidence').text,
            "narrative": log_element.find('narrative').text,
        }
    return TOPICS


def load_qrels(qrels_filepath):
    with open(qrels_filepath) as f:
        reader = csv.reader(f, delimiter=' ')
        return list(reader)


def write_to_file(lines, output):
    with open(output, "w") as f:
        print(*lines, sep="\n", file=f)


def add_gain(qrels):
    """
    Adds column with gain values.

    gain = 4: useful, correct, credible.
    gain = 3: useful, correct, not credible or no judgment.
    gain = 2: useful, no answer or no judgment, credible.
    gain = 1: useful, no answer or no judgment, not credible or no judgment.
    gain = 0: not useful.
    gain = -1: useful, incorrect, not credible or no judgment.
    gain = -2: useful, incorrect, credible.
    """
    lines = []
    for line in qrels:
        topic, _, docid, useful, answer, credibility = line
        topic_answer = TOPICS[int(topic)]["answer"]
        binary_topic_answer = 1 if topic_answer == "yes" else 0
        useful = 1 if useful == "1" else 0
        credibility = int(credibility)
        answer = int(answer)
        correct   = 1 if (useful == 1 and ((answer == 1 and topic_answer == "yes") or (answer == 2 and topic_answer == "no" ))) else 0
        incorrect = 1 if (useful == 1 and ((answer == 1 and topic_answer == "no" ) or (answer == 2 and topic_answer == "yes"))) else 0

        gain = None
        if   useful == 1 and correct == 1 and credibility == 1:
            gain = 4
        elif useful == 1 and correct == 1 and credibility <= 0:
            gain = 3
        elif useful == 1 and answer <= 0  and credibility == 1:
            gain = 2
        elif useful == 1 and answer <= 0  and credibility <= 0:
            gain = 1
        elif useful == 0:
            gain = 0
        elif useful == 1 and incorrect == 1 and credibility <= 0:
            gain = -1
        elif useful == 1 and incorrect == 1 and credibility == 1:
            gain = -2
        else:
            raise Exception( 'did not assign a gain value')
                          
        lines.append(f"{topic}{OUTPUT_SEPARATOR}"
                     f"{_}{OUTPUT_SEPARATOR}"
                     f"{docid}{OUTPUT_SEPARATOR}"
                     f"{gain}")
    return lines


def binary_useful(qrels):
    lines = []
    for line in qrels:
        topic, _, docid, useful, answer, credibility = line
        lines.append(f"{topic}{OUTPUT_SEPARATOR}"
                     f"{_}{OUTPUT_SEPARATOR}"
                     f"{docid}{OUTPUT_SEPARATOR}"
                     f"{useful}")
    return lines


def binary_useful_correct_credible(qrels):
    lines = []
    for line in qrels:
        topic, _, docid, useful, answer, credibility = line
        topic_answer = TOPICS[int(topic)]["answer"]
        binary_topic_answer = 1 if topic_answer == "yes" else 0
        useful = 1 if useful == "1" else 0
        credibility = 1 if credibility == "1" else 0
        answer = int(answer)
        correct = 1 if (answer == 1 and binary_topic_answer == 1) or (answer == 2 and binary_topic_answer == 0) else 0

        column_value = useful & correct & credibility
        lines.append(f"{topic}{OUTPUT_SEPARATOR}"
                     f"{_}{OUTPUT_SEPARATOR}"
                     f"{docid}{OUTPUT_SEPARATOR}"
                     f"{column_value}")
    return lines


def binary_useful_credible(qrels):
    lines = []
    for line in qrels:
        topic, _, docid, useful, answer, credibility = line
        useful = 1 if useful == "1" else 0
        credibility = 1 if credibility == "1" else 0
        column_value = useful & credibility
        lines.append(f"{topic}{OUTPUT_SEPARATOR}"
                     f"{_}{OUTPUT_SEPARATOR}"
                     f"{docid}{OUTPUT_SEPARATOR}"
                     f"{column_value}")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--topics', '-t', required=True, type=Path,
                        help='Topics file')
    parser.add_argument('--qrels', '-q', required=True, type=Path,
                        help='Qrels file')
    parser.add_argument('--space_separated', '-s', action="store_true",
                        help='Changes type of separator in output from tab to space')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', '-o', type=Path, help='output filename')
    group.add_argument('--print', '-p', action='store_true', help='print to console')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--v1', action="store_true",
                       help="""
                            Creates gain value column with following values:
                            gain = 4: useful, correct, credible.
                            gain = 3: useful, correct, not credible or no judgment.
                            gain = 2: useful, no answer or no judgment, credible.
                            gain = 1: useful, no answer or no judgment, not credible or no judgment.
                            gain = 0: not useful.
                            gain = -1: useful, incorrect, not credible or no judgment.
                            gain = -2: useful, incorrect, credible.
                            """)
    group.add_argument('--v2', action="store_true",
                       help="Keeps `Useful` values only (binary format)")
    group.add_argument('--v3', action="store_true",
                       help="Adds a binary column with 1 as useful and correct and "
                            "credible, 0 otherwise")
    group.add_argument('--v4', action="store_true",
                       help="Adds a binary column with 1 as useful and credible "
                            ", 0 otherwise")
    args = parser.parse_args()

    if args.space_separated:
        global OUTPUT_SEPARATOR
        OUTPUT_SEPARATOR = " "

    load_topics(args.topics)
    qrels = load_qrels(args.qrels)

    lines = []
    if args.v1:
        lines = add_gain(qrels)
    elif args.v2:
        lines = binary_useful(qrels)
    elif args.v3:
        lines = binary_useful_correct_credible(qrels)
    elif args.v4:
        lines = binary_useful_credible(qrels)

    if args.print:
        print(*lines, sep="\n")
    elif args.output:
        write_to_file(lines, args.output)


if __name__ == '__main__':
    main()


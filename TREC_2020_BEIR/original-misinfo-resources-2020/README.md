# Original Repository

This is a copy as of Nov 1, 2020 of the repository at https://github.com/trec-health-misinfo/resources .  This is a private repository because it contains qrels.  The track's qrels are to remain private until the release of the official proceedings.  The qrels are stored at NIST in a password secured location for 2020 track participants: https://trec.nist.gov/act_part/tracks2020.html .

# Resources

This repository contains code and data for the [TREC Decision and Health Misinformation Track](https://trec-health-misinfo.github.io/).

# 2020 Resources

## Evaluation

Runs are evaluated by using an [extension of trec_eval](https://github.com/trec-health-misinfo/Trec_eval_extension) to handle multiple aspects as well the [compatibility](https://github.com/trec-health-misinfo/Compatibility) measure.

In all cases, we derive a qrels file to use with the specific measure from the original qrels file.  Descriptions of the derived qrels files are given below.

| derived qrels                                | measure       | run type | notes           |
| -------------------------------------------- | ------------- | -------- | --------------- |
| misinfo-qrels-graded.helpful-only            | compatibility | adhoc    | Want high score. Use together with harmful. |
| misinfo-qrels-graded.harmful-only            | compatibility | adhoc    | Want low score. Use together with helpful.  |
| misinfo-qrels-binary.useful                  | ndcg          | adhoc    | Effectively "topic relevance".              |
| misinfo-qrels-binary.useful-correct          | ndcg          | adhoc    | Can only be correct if useful.              |
| misinfo-qrels-binary.useful-credible         | ndcg          | adhoc    | Can only be credible if useful.             |
| misinfo-qrels-binary.useful-correct-credible | ndcg          | adhoc    |                                             |
| misinfo-qrels.2aspects.correct-credible      | cam_map       | adhoc    |                                             |
| misinfo-qrels.2aspects.useful-credible       | cam_map       | adhoc    |                                             |
| misinfo-qrels.3aspects                       | cam_map_three | adhoc    |                                             |
| misinfo-qrels-binary.incorrect               | Rprec         | recall   | !correct does not neccessarily equal incorrect, see below |

Summaries for runs are produced using the scripts/run-2020-eval.sh file, which shows how to run the evaluation.

## Topics

File: topics/misinfo-2020-topics.xml is a copy of https://trec-health-misinfo.github.io/topics.xml

## qrels (relevance judgments)

NIST used the [track relevance assessing guidelines](https://trec-health-misinfo.github.io/docs/AssessingGuidelines-2020.pdf) to generate the track's qrels.

File: qrels/misinfo-2020-qrels

Format:  
column 1: topic id  
column 2: ignore  
column 3: docno  
column 4: useful = 1, not useful = 0  
column 5: is whether the document answers the question:  Doesn't Answer (0), Answers Yes (1), Answers No (2), and -1 means no judgment  
column 6: is Credibility: Credible(1), Not Credible (0) where -1 is no judgment  

Notes: 

A. When a document was judged as not useful, it was not judged for its answer nor for its credibility.  In some cases, a useful document was accidentally not judged for its answer or credibility, i.e. a "skip".  

B. Some participants submitted docids that were not WARC doc types of "response".  While not explained clearly in the track guidelines, only WARC records of type "response" should have been used.  In almost all cases, when an assessor was given a non-response docid to judge, it was judged "not useful".  Rather than confuse matters by including non-response documents in the qrels, these qrels contain only the judgements for docs of type "response".

C. There are four missing topics: 33, 35, 36, and 48.  NIST ran out of time and was not able to judge all topics.

D. Topics 9 and 17 are duplicate topics, but have different judgments.  We accidentally duplicated the topics.

E. The 2020 Health Misinformation qrels, file: misinfo-2020-qrels, were named misinfo-qrels.fixed.response-only during development.

### Derived qrels

We use the misinfo-2020-qrels to derive special qrel files to allow us to evaluate the adhoc and recall runs.  The derived qrels are generated with scripts/gen-2020-dervied-qrels.sh bash script.  The derived qrels are found in qrels/2020-derived-qrels .

For the derived qrels, we only include "relevant" documents, for not all topics contain results that meet the success criteria of a "relevant" document in the derived qrels.  By excluding topics without "relevant" documents, the effectiveness measures are only computed over topics for which runs could feasible get a non-zero score.

#### Graded / Basic preference levels 

We can convert the 3 aspects judged for documents into a basic preference ordering or graded relevance values.  A document is correct if it contains an answer that matches the topic's given answer.   For example, topic 1's given answer is "no".  If a document for topic 1 has an answer of "no", then it is correct. An incorrect document if it contains an incorrect answer.  For a document to be correct or incorrect, it has to be useful.

```python
     correct   = 1 if (useful == 1 and ((answer == 1 and topic_answer == "yes") or (answer == 2 and topic_answer == "no" ))) else 0
     incorrect = 1 if (useful == 1 and ((answer == 1 and topic_answer == "no" ) or (answer == 2 and topic_answer == "yes"))) else 0
```

Note: "not correct" does not necessarily mean "incorrect".  A "not useful" document is neither correct nor incorrect, for a "not useful" document is off topic.  Likewise, if a document doesn't contain an answer or is not judged for "answer", it's neither correct nor incorrect.

score = 4 , useful, correct, credible

score = 3 , useful, correct, not credible or no credibility judgment

score = 2, useful, doesn't answer or no judgment for answer, credible

score = 1, useful, doesn't answer or no judgment for answer, not credible or no credibility judgment

score = 0 , not useful.  Ignore answer and credibility.

score = -1, useful, incorrect, not credible or no credibility judgment

score = -2, useful, incorrect, credible

It is tempting to use the above scores to compute ndcg, but that ignores the incorrect information.  A better solution is to create a set of helpful and harmful qrels:

```awk
gawk '$4 > 0 {print $1, $2, $3, $4}' < misinfo-qrels-graded > misinfo-qrels-graded.helpful-only
gawk '$4 < 0 {print $1, $2, $3, ($4 * -1)}' < misinfo-qrels-graded > misinfo-qrels-graded.harmful-only
```

Then, we can use them with [compatibility](https://github.com/claclark/Compatibility).  We also have a [modifed version of compatibility](https://github.com/trec-health-misinfo/Compatibility) that outputs in trec_eval format.

#### Binary Relevance

We created a series of files in the standard qrels format for binary relevance effectiveness measures.  For these files, a document is correct if it contains an answer that matches the topic's given answer.  For example, topic 1's given answer is "no".  If a document for topic 1 has an answer of "no", then it is correct. If a document "doesn't answer" or was not judged for an answer, then it is not correct.  A document is credible if only judged to be credible, otherwise it is not credible.

file: misinfo-qrels-binary.useful

Same as qrels/misinfo-2020-qrels, but without columns 5 (answer) and 6 (credibility)

file: misinfo-qrels-binary.useful-correct-credible

The document is judged as "relevant" if it was useful and correct and credible.  

file: misinfo-qrels-binary.useful-credible

The document is judged as "relevant" if it was useful and credible. Ignores correctness.

file: misinfo-qrels-binary.useful-correct

The document is judged as "relevant" if it was useful and correct. Ignores credibility. 

file: misinfo-qrels-binary.incorrect

A document is "relevant" if it is useful and has an incorrect answer.  In terms of qrels/misinfo-2020-qrels, "relevance" is mapped as follows:

incorrect = 0 if column 4 == 0  
incorrect = 0 if column 4 == 1 && (column 5 == 0 || column5 == -1)   
incorrect = 1 if column 4 == 1 && column 5 == 2 && topic.answer == "yes"   
incorrect = 1 if column 4 == 1 && column 5 == 1 && topic.answer == "no"  

Note: because documents can only be correct and credible when they are useful (as per the relevance assesing guidelines), files representing the 3 separate aspects are:

1. misinfo-qrels-binary.useful
2. misinfo-qrels-binary.useful-correct
3. misinfo-qrels-binary.useful-credible

#### Multiple Aspect

We created both two aspect and three aspect qrel files for use with the extended trec_eval : https://github.com/lcschv/Trec_eval_extension .

file:  misinfo-qrels.3aspects

Same as the qrels/misinfo-2020-qrels file, but with column 5 (answer) and column 6 (credibility) mapped as follows:

column 5:  
0 -> 0  
1 -> 1 if topic.answer is "yes" else -> 0  
2 -> 1 if topic.answer is "no" else -> 0  
-1 -> 0  

column 6:  
1 -> 1  
0 -> 0  
-1 -> 0  

file:  misinfo-qrels.2aspects.useful-credibile

Same as  misinfo-qrels.3aspects except column 5 (answer) is removed.  So, in this file, column 5 is credibility.

file: misinfo-qrels.2aspects.correct-credible

Column 4 is 1 if document is useful and correct (columns 4 and 5 in misinfo-qrels.3aspects).  Column 5 is same as column 6 in misinfo-qrels.3aspects (credibility).



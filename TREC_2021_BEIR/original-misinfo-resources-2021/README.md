# Resources

This repository contains code and data for the [TREC Decision and
Health Misinformation Track](https://trec-health-misinfo.github.io/).

# 2021 Resources

## Evaluation

Runs are evaluated by using an [extension of
trec_eval](https://github.com/trec-health-misinfo/Trec_eval_extension)
to handle multiple aspects as well the
[compatibility](https://github.com/trec-health-misinfo/Compatibility)
measure.

In all cases, we derive a qrels file to use with the specific measure
from the original qrels file.  Descriptions of the derived qrels files
are given below.

| derived qrels                                | measure       |  notes           |
| -------------------------------------------- | ------------- |  --------------- |
| misinfo-qrels-graded.helpful-only            | compatibility |  Want high score. Use together with harmful.   |
| misinfo-qrels-graded.harmful-only            | compatibility |  Want low score. Use together with helpful.    |
| misinfo-qrels-graded.usefulness              | ndcg          |  The usefulness of the document alone.         |
| misinfo-qrels-binary.useful-correct          | ndcg          |  Can only be correct if useful. See also P@10. |
| misinfo-qrels-binary.useful-correct          | P@10          |  Can only be correct if useful. See also ndcg. |
| misinfo-qrels-binary.useful-credible         | ndcg          |  Can only be credible if useful.               |
| misinfo-qrels-binary.useful-correct-credible | ndcg          |                                                |
| misinfo-qrels.2aspects.correct-credible      | cam_map       |                                                |
| misinfo-qrels.2aspects.useful-credible       | cam_map       |                                                |
| misinfo-qrels.3aspects                       | cam_map_three |                                                |
| misinfo-qrels-binary.incorrect               | P@10          | !correct does not neccessarily equal incorrect, see below |

Summaries for runs are produced using the scripts/run-2021-eval.sh
file, which shows how to run the evaluation.

## Topics

File: topics/misinfo-2021-topics.xml is a copy of
https://trec.nist.gov/act_part/tracks/misinfo/misinfo-2021-topics.xml

## qrels (relevance judgments)

NIST used the track's [relevance assessing
guidelines](https://trec-health-misinfo.github.io/docs/TREC-2021-Health-Misinformation-Track-Assessing-Guidelines_Version-2.pdf)
to generate the track's qrels.

File: qrels/qrels-35topics.txt

Format:  
+ column 1: topic id  
+ column 2: ignore  
+ column 3: docno  
+ column 4: usefulness (see below)  
+ column 5: supportiveness (see below) 
+ column 6: credibility (see below)

Usefulness:
+    0 "Not Useful"
+    1 "Useful"
+    2 "Very Useful"

Supportiveness:
+   -2 "Dimension inadvertently not judged"
+   -1 "Document judged not useful" -> means supportiveness not judged on purpose
+    0 "Dissuades"
+    1 "Neutral"
+    2 "Supportive"

For topics with a helpful stance, supportive docs are better than
neutral, which are better than dissuades.

For topics with an unhelpful stance, dissuades are better than
neutral, which are better than supportive.

Credibility:
+   -2 "Dimension inadvertently not judged"
+   -1 "Document judged not useful"
+    0 "Low"
+    1 "Good"
+    2 "Excellent"

Notes: 

A. When a document was judged as not useful, it was not judged for
supportiveness nor for credibility.  In some cases, a useful document
was accidentally not judged for its supportiveness or credibility.

B. Only 35 topics were judged. NIST was not able to judge all topics.

### Derived qrels

We use the qrels/qrels-35topics.txt to derive special qrels files to
allow us to evaluate the runs.  The derived qrels are generated with
scripts/gen-2021-dervied-qrels.sh bash script.  The derived qrels are
found in qrels/2022-derived-qrels/ .

For these derived qrels:
+ misinfo-qrels-binary.useful-correct-credible
+ misinfo-qrels-binary.useful-credible
+ misinfo-qrels-binary.useful-correct
+ misinfo-qrels.3aspects
+ misinfo-qrels.2aspects.useful-credible
+ misinfo-qrels-binary.incorrect

we only include "relevant/useful" documents, for not all topics
contain results that meet the success criteria of a "relevant"
document in the derived qrels.  By excluding topics without "relevant"
documents, the effectiveness measures are only computed over topics
for which runs could feasibly get a non-zero score.  Note: In 2020, we
mistakenly included the misinfo-qrels.2aspects.correct-credible
dervied qrels in this set.

For the derived qrels misinfo-qrels.2aspects.correct-credible, we only
retain the doc if it is at least correct or credible (or both).

#### Graded / Basic preference levels 

We can convert the 3 aspects judged for documents into a basic
preference ordering or graded relevance values.  For topics with a
"helpful" stance:

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

For unhelpful topics:

score = 12: very useful, dissuades, excellent  
score = 11: useful, dissuades, excellent   
score = 10: very useful, dissuades, good  
score = 9: useful, dissuades, good  
score = 8: very useful, dissuades, low or not judged  
score = 7: useful, dissuades, low or not judged  
score = 6: very useful, neutral or not judged, excellent  
score = 5: useful, neutral or not judged , excellent   
score = 4: very useful, neutral or not judged , good  
score = 3: useful, neutral or not judged , good  
score = 2: very useful, neutral or not judged   , low or not judged  
score = 1: useful, neutral or not judged , low or not judged  
score = 0: not useful (should not be judged for supportiveness nor for credibility)  
score = -1: very useful or useful, supportive, low or not judged  
score = -2: very useful or useful, supportive, good  
score = -3: very useful or useful, supportive, excellent  

It is tempting to use the above scores to compute ndcg, but that
ignores the incorrect information.  A better solution is to create a
set of helpful and harmful qrels:

```awk
gawk '$4 > 0 {print $1, $2, $3, $4}' < misinfo-qrels-graded > misinfo-qrels-graded.helpful-only
gawk '$4 < 0 {print $1, $2, $3, ($4 * -1)}' < misinfo-qrels-graded > misinfo-qrels-graded.harmful-only
```

Then, we can use them with
[compatibility](https://github.com/claclark/Compatibility).  For the
track, we used a [modifed version of
compatibility](https://github.com/trec-health-misinfo/Compatibility)
that outputs in trec_eval format.

#### Binary Relevance

We created a series of files in the standard qrels format for binary
relevance effectiveness measures.  The use of these files is limited, and
we recommend the use of the misinfo-qrels-graded.helpful-only and
misinfo-qrels-graded.harmful-only qrels for evaluation.

For binary relevance of usefulness, if usefulness > 0, then it is
useful (relevant).

For the binary relevance files, a document is correct if it is
supportive of helpful treatments and dissuades unhelpful treatments.
A document is incorrect if it dissuades from helpful treatments and is
supportive of unhelpful treatments.  Note that neutral documents are
neither correct nor incorrect.

A document is credible if only judged to have good or high credibility
(i.e. credibility > 0), otherwise it is not credible.

file: misinfo-qrels-binary.useful

The document is "relevant" if usefulness > 0.

file: misinfo-qrels-binary.useful-correct-credible

The document is judged as "relevant" if it was useful and correct and credible.  

file: misinfo-qrels-binary.useful-credible

The document is judged as "relevant" if it was useful and
credible. Ignores correctness.

file: misinfo-qrels-binary.useful-correct

The document is judged as "relevant" if it was useful and
correct. Ignores credibility.

file: misinfo-qrels-binary.incorrect

A document is "relevant" if it is useful and has an incorrect answer.  

Note: because documents can only be correct and credible when they are
useful (as per the relevance assesing guidelines), files representing
the 3 separate aspects are:

1. misinfo-qrels-binary.useful
2. misinfo-qrels-binary.useful-correct
3. misinfo-qrels-binary.useful-credible

#### Multiple Aspect

We created both two aspect and three aspect qrel files for use with
the extended trec_eval : https://github.com/lcschv/Trec_eval_extension
.

file:  misinfo-qrels.3aspects

Same as the qrels/35-topics.txt, but with column 5 (supportiveness)
mapped to correctness, and column 6 (credibility) mapped such that
negative values become 0.

file:  misinfo-qrels.2aspects.useful-credibile

Same as misinfo-qrels.3aspects except column 5 is removed.  So, in
this file, column 5 is credibility.

file: misinfo-qrels.2aspects.correct-credible

Column 4 is 1 if document is useful and correct (columns 4 and 5 in
misinfo-qrels.3aspects).  Column 5 is same as column 6 in
misinfo-qrels.3aspects (credibility).


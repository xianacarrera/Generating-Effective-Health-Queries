# Resources

This repository contains code and data for the [TREC 2022 Decision and Health Misinformation Track](https://trec-health-misinfo.github.io/).

# 2022 Resources

## Evaluation

Answer Prediction runs are evaluated using classical classification evaluation metrics: True Positive Rate, False Positive Rate, Accuracy, and AUC (Area Under the Curve).  The answers for each topic are found in topics/misinfo-2022-topics.xml.

Web Retrieval runs are evaluated by using an [extension of trec_eval](https://github.com/trec-health-misinfo/Trec_eval_extension) to handle multiple aspects as well the [compatibility](https://github.com/trec-health-misinfo/Compatibility) measure.  To evaluate the web retrieval runs, we derive a qrels file to use with the specific measure from the original qrels file and preference judgments. Descriptions of the derived qrels files are given below.

| derived qrels                                | measure       |  notes           |
| -------------------------------------------- | ------------- |  --------------- |
| misinfo-qrels-graded.helpful-only            | compatibility |  Want high score. Use together with harmful.   |
| misinfo-qrels-graded.harmful-only            | compatibility |  Want low score. Use together with helpful.    |
| misinfo-qrels-graded.usefulness              | ndcg          |  The usefulness of the document alone. **BEWARE: Ignores correctness!** |
| misinfo-qrels-binary.useful-correct          | ndcg          |  Can only be correct if useful. See also P@10. |
| misinfo-qrels-binary.useful-correct          | P@10          |  Can only be correct if useful. See also ndcg. |
| misinfo-qrels-binary.incorrect               | P@10          | !correct does not neccessarily equal incorrect, see below |

Summaries for runs are produced using the scripts/run-2022-eval.sh file, which shows how to run the evaluation.

## Topics

File: topics/misinfo-2022-topics.xml is a copy of https://trec.nist.gov/act_part/tracks/misinfo/misinfo-2022-topics.xml

## qrels (relevance judgments)

This year, we have two phases of judging: in the first phase, assessors were asked to judge in terms of relevance and document answers (assessors' perceived answers to the topic question after reading the document); after the first phase, (very) useful and correct (aligned with the topic's provided answer) documents were passed to assessors to perform preference judging. For most topics, only very useful and correct documents were preference judged.  Preference judging continued until the top 10 documents were determined.  If fewer than 10 documents were passed to phase 2, only those documents were preference judged. Due to the time limit, assessors didn't finish preference judging for all topics, and the organizers completed the preference judging.  

NIST used the track's [relevance assessing guidelines](https://trec-health-misinfo.github.io/docs/TREC-2022-Health-Misinformation-Track-Assessing-Guidelines.pdf) to generate the judgments in qrels/qrels.final.oct-19-2022 and the preferences in qrels/trec2022_act26_v2.csv.  The qrels have the following format:

<!-- From Ian in Slack: 
Here are qrels for six completed topics.  The format is
topic docid useful answer
where useful is -1, 0=not, 1=useful, 2=very, and
where answer is -1, 0=no, 1=yes, 2=unclear
-->

Format:  
+ column 1: topic id  
+ column 2: docno  
+ column 3: usefulness (see below)  
+ column 4: answer (see below) 

Usefulness:
+    -1 "Not Judged / Error"
+    0 "Not Useful"
+    1 "Useful"
+    2 "Very Useful"

Document's Answer:
+   -1 "Document judged not useful" -> means answer not judged on purpose
+    0 "No"
+    1 "Yes"
+    2 "Unclear"

For topics with a "yes" answer, "yes" docs are better than "unclear", which are better than "no".

For topics with a "no" answer, "no" docs are better than "unclear", which are better than "yes".

The preferences file, rels/trec2022_act26_v2.csv, has the following columns in CSV format:

+ ID : The assessor who did the preference judging.
+ Topic ID : The question and answer shown to the assessor.
+ Is Completed : Was judging completed? TRUE for all topics in file.  Ignore this column.
+ Grade : The preference level.  1 is the most preferred.  Multiple documents can be at the same preference level.
+ Document UUID : This is the docno.
+ Task : Internal ID.  This is NOT the topic number.  Ignore this column.

To join the preference data with qrels data, a mapping between the "Topic ID" column and the topic number must be made. We perform this join in the gen-qrels-for-compatibility.py script.

Notes: 

A. When a document was judged as not useful, it was not judged for answers.

B. In the first phase, only 45 topics were judged. In the second phase, only 38 topics required further preference judging. For example, on topic 152, all "very useful" documents had a document answer of "yes" or "unclear", but the correct answer was "no".  Thus, for topic 152, no preference judging was completed.  For most topics, we only preference judged the "very useful" and correct documents.

### Derived qrels

We use the qrels/qrels.final.oct-19-2022 and qrels/trec2022_act26_v2.csv to derive special qrels files to allow us to evaluate the runs.
The derived qrels are generated with scripts/gen-qrels-for-compatibility.py Python script.  
The derived qrels are found in qrels/2022-derived-qrels/.

For these derived qrels:
+ misinfo-qrels-binary.useful-correct
+ misinfo-qrels-binary.incorrect

we only include "relevant/useful" documents, for not all topics contain results that meet the success criteria of a "relevant" document in the derived qrels.  
By excluding topics without "relevant" documents, the effectiveness measures are only computed over topics for which runs could feasibly get a non-zero score. 

#### Graded / Basic preference levels 

We can convert the 2 aspects judged for documents into a basic preference ordering or graded relevance values.  

For topics with a "yes" answer:

+ score = 4: very useful, yes
+ score = 3: useful, yes
+ score = 2: very useful, unclear or not judged
+ score = 1: useful, unclear or not judged
+ score = 0: not useful, not judged
+ score = -1: useful, no
+ score = -2: very useful, no  

For topics with a "no" answer:

+ score = 4: very useful, no
+ score = 3: useful, no
+ score = 2: very useful, unclear or not judged
+ score = 1: useful, unclear or not judged
+ score = 0: not useful, not judged
+ score = -1: useful, yes
+ score = -2: very useful, yes

Then, for topics with preference judgments from the second phase, we put them with higher positions over the basic preference ordering. 
For example, if a topic has ten preference judgments, the most preferred document (which is judged as "very useful, correct" in the first phase) will have a score of 4 + 10 + 1 = 15.

It is tempting to use the above scores to compute ndcg, but that ignores the incorrect information.  A better solution is to create a set of helpful and harmful qrels:

```awk
gawk '$4 > 0 {print $1, $2, $3, $4}' < misinfo-qrels-graded > misinfo-qrels-graded.helpful-only
gawk '$4 < 0 {print $1, $2, $3, ($4 * -1)}' < misinfo-qrels-graded > misinfo-qrels-graded.harmful-only
```

Then, we can use them with [compatibility](https://github.com/claclark/Compatibility).  
For the track, we used a [modifed version of compatibility](https://github.com/trec-health-misinfo/Compatibility) that outputs in trec_eval format.

Please note that because we remove non-relevant documents from misinfo-qrels-graded.helpful-only and misinfo-qrels-graded.harmful-only, not all topics are necessarily listed in these files.

#### Binary Relevance

We created a series of files in the standard qrels format for binary relevance effectiveness measures.  
The use of these files is limited, and we recommend the use of the misinfo-qrels-graded.helpful-only and misinfo-qrels-graded.harmful-only qrels for evaluation.

For binary relevance of usefulness, if usefulness > 0, then it is useful (relevant).

For the binary relevance files, a document is correct if the document's answer judgment is "yes" when the topic's answer is "yes", or the document's answer judgment is "no" when the topic's answer is "no".
A document is incorrect if the document's answer judgment is "yes" when the topic's answer is "no", or the document's answer judgment is "no" when the topic's answer is "yes".
Note that "unclear" or "not judged" documents are neither correct nor incorrect.

file: misinfo-qrels-binary.useful

The document is "relevant" if usefulness > 0.

file: misinfo-qrels-binary.useful-correct

The document is judged as "relevant" if it was useful and correct.

file: misinfo-qrels-binary.incorrect

A document is "relevant" if it is useful and has an incorrect answer.  

# Producing Evaluation scores

The run summaries that were distributed to participants were created using the scripts/run-eval.sh script.  Please see this script to understand how to use the derived qrels.

Note that the organizers compute the difference between helpful and harmful compatibility for only topics that have both helpful and harmful documents.  Please see the overview for details.


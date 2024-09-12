
# Required Packages `pandas`
# Install using `pip install pandas`

# If using Compute Canada, you can use the command `module load scipy-stack`
# to load an interpreter with pandas pre-installed.

# usage:
#
# run from scripts directory
#
# bash gen-2021-derived-qrels.sh
#

QRELS="../qrels/qrels-35topics.txt"
TOPICS="../topics/misinfo-2021-topics.xml"
DERIVEDDIR="../qrels/2021-derived-qrels"

python3 ./gen_derived_qrels.py --qrels $QRELS  --topics $TOPICS  --output $DERIVEDDIR

cd $DERIVEDDIR

# the gains file does not make sense on its own, for the incorrect docs DO have an effect
rm -f misinfo-qrels-graded 

# we have graded usefulness, we don't need binary judgments:
rm -f misinfo-qrels-binary.useful

# scripts also generate some summary data, these are not qrels
rm -f misinfo-qrels.counts.txt

# The NIST assessors do not judge supportiveness nor credibility unless useful > 0.
# To avoid confusion with these derived qrels, we only keep docs judged useful.
# Likewise, we do this for the binary.correct and binary.incorrect to avoid
# making the mistake that !correct equals incorrect.  See README.md .
# We do not change misinfo-qrels-graded.usefulness nor misinfo-qrels-binary.useful,
# for they make sense as is.
#
# Because we're deriving these qrels, it is possible for us to have topics
# with no relevant documents.  Rather than assign a score of zero to a retrieval
# measure in these cases, it is better to produce no result and correctly reflect
# that we didn't have a proper topic for this measure.  So, by removing these
# docs with a usefulness of 0, we make sure that if a topic gets a evaluation
# score, it means there was something there to evaluate it.
#
# In 2020, we mistakenly included misinfo-qrels.2aspects.correct-credible in this
# stripping of judgments.  


FILES=(misinfo-qrels-binary.useful-correct-credible misinfo-qrels-binary.useful-credible misinfo-qrels-binary.useful-correct misinfo-qrels.3aspects misinfo-qrels.2aspects.useful-credible misinfo-qrels-binary.incorrect)

for f in ${FILES[@]}; do
    mv $f "$f.tmp"
    gawk '$4 > 0 {print}' < "$f.tmp" > $f
    rm -f "$f.tmp"
done

# only keep doc if correct or credible (cannot be incorrect and non-credible)
mv misinfo-qrels.2aspects.correct-credible misinfo-qrels.2aspects.correct-credible.tmp
gawk '$4 > 0 || $5 > 0 {print}' < misinfo-qrels.2aspects.correct-credible.tmp > misinfo-qrels.2aspects.correct-credible
rm -f misinfo-qrels.2aspects.correct-credible.tmp








# usage:
#
# run from scripts directory
#
# bash gen-2020-derived-qrels.sh
#

cd ../qrels/2020-derived-qrels

QRELS="../misinfo-2020-qrels"
TOPICS="../../topics/misinfo-2020-topics.xml"

# File: misinfo-qrels-graded
#
# Map to graded rel as following:
# 
# gain = 4 , useful, correct, credible
# gain = 3 , useful, correct, not credible or no judgment
# gain = 2, useful, no answer or no judgment, credible
# gain = 1, useful, no answer or no judgment, not credible or no judgment
# gain = 0 , not useful
# gain = -1, useful, incorrect, not credible or no judgment
# gain = -2, useful, incorrect, credible
# Output would be a new qrels file named "misinfo-qrels-graded" with format (no header):
python3 ../../scripts/format.py --topics $TOPICS --qrels $QRELS --v1 --output misinfo-qrels-graded

gawk '$4 > 0 {print $1, $2, $3, $4}' < misinfo-qrels-graded > misinfo-qrels-graded.helpful-only
gawk '$4 < 0 {print $1, $2, $3, ($4 * -1)}' < misinfo-qrels-graded > misinfo-qrels-graded.harmful-only
# the gains file does not make sense on its own, for the incorrect docs DO have an effect
rm -f misinfo-qrels-graded 

# File: misinfo-qrels-binary.useful
# strip off columns 5 and 6 of qrels
python3 ../../scripts/format.py --topics $TOPICS --qrels $QRELS --v2 --output misinfo-qrels-binary.useful

# File: misinfo-qrels-binary.useful-correct-credible
# useful and correct and credible
python3 ../../scripts/format.py --topics $TOPICS --qrels $QRELS --v3 --output misinfo-qrels-binary.useful-correct-credible

# File: misinfo-qrels-binary.useful-credible
python3 ../../scripts/format.py --topics $TOPICS --qrels $QRELS --v4 --output misinfo-qrels-binary.useful-credible

# File: misinfo-qrels-binary.useful-correct
# File: misinfo-qrels.3aspects
# File: misinfo-qrels.2aspects.correct-credible
# File: misinfo-qrels.2aspects.useful-credible
# File: misinfo-qrels-binary.incorrect

python3 ../../scripts/script_files_6_11.py --qrels $QRELS --topic $TOPICS --output .

# scripts also generate some summary data, these are not qrels
rm -f misinfo-qrels.counts.txt misinfo-qrels.for-R.txt

FILES=(misinfo-qrels-binary.useful misinfo-qrels-binary.useful-correct-credible misinfo-qrels-binary.useful-credible misinfo-qrels-binary.useful-correct misinfo-qrels.3aspects misinfo-qrels.2aspects.useful-credible misinfo-qrels.2aspects.correct-credible misinfo-qrels-binary.incorrect)

for f in ${FILES[@]}; do
    mv $f "$f.tmp"
    gawk '$4 > 0 {print}' < "$f.tmp" > $f
    rm -f "$f.tmp"
done








# To run this script, you need to set all of these paths correctly
# and you need to make the output directories.
ADHOC_OUT_DIR="/home/smucker/project/smucker/group-data/trec/health-misinfo-2020/run-summaries/adhoc"
RECALL_OUT_DIR="/home/smucker/project/smucker/group-data/trec/health-misinfo-2020/run-summaries/recall"

# directories with the runs to evaluate
ADHOC="/home/smucker/project/smucker/group-data/trec/restricted-access/health-misinfo-2020/misinfo-runs/adhoc"
RECALL="/home/smucker/project/smucker/group-data/trec/restricted-access/health-misinfo-2020/misinfo-runs/recall"

QRELS="/home/smucker/git-repos/trec-misinfo-resources/qrels/2020-derived-qrels"

# extended trec_eval: https://github.com/lcschv/Trec_eval_extension
trec_eval="/home/smucker/git-repos/Trec_eval_extension/Trec_eval_extension/trec_eval"
# https://github.com/trec-health-misinfo/Compatibility
compatibility="/home/smucker/git-repos/Compatibility/compatibility.py"

############ Config section above ends.  You should not need to edit anything below here. ###############

if [ ! -d $ADHOC_OUT_DIR ]
then
    echo "Adhoc runs summaries directory named \"$ADHOC_OUT_DIR\" does not exist."
    exit
fi

if [ ! -d $RECALL_OUT_DIR ]
then
    echo "Recall runs summaries directory named \"$RECALL_OUT_DIR\" does not exist."
    exit
fi

if [ ! -d $ADHOC ]
then
    echo "Adhoc runs input directory named \"$ADHOC\" does not exist."
    exit
fi

if [ ! -d $RECALL ]
then
    echo "Recall runs input directory named \"$RECALL\" does not exist."
    exit
fi

if [ ! -d $QRELS ]
then
    echo "Derived qrels directory named \"$QRELS\" does not exist."
    exit
fi

if [ ! -f $trec_eval ]
then
    echo "Extended trec_eval at \"$trec_eval\" does not exist."
    exit
fi

if [ ! -f $compatibility ]
then
    echo "Python script \"$compatibility\" does not exist."
    exit
fi

# misinfo-qrels.2aspects.correct-credible	cam_map	adhoc
# misinfo-qrels.2aspects.useful-credible	cam_map	adhoc
# misinfo-qrels.3aspects	                cam_map_three	adhoc
# misinfo-qrels-binary.incorrect	        Rprec	recall
# misinfo-qrels-binary.useful	                ndcg	adhoc
# misinfo-qrels-binary.useful-correct	        ndcg	adhoc
# misinfo-qrels-binary.useful-correct-credible	ndcg	adhoc
# misinfo-qrels-binary.useful-credible	        ndcg	adhoc
# misinfo-qrels-graded.harmful-only	        compatibility	adhoc
# misinfo-qrels-graded.helpful-only	        compatibility	adhoc

# Ad-hoc runs


for RUN_FULL_PATH in $ADHOC/*
do
    RUN_NAME=`basename $RUN_FULL_PATH`
    SUMMARY="$ADHOC_OUT_DIR/$RUN_NAME.summary"
    printf "run\tqrels\tmeasure\ttopic\tscore\n" > $SUMMARY
    
    $trec_eval -q -c -M 1000 -m cam_map -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.correct-credible $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.correct-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

    $trec_eval -q -c -M 1000 -m cam_map -R qrels_twoaspects $QRELS/misinfo-qrels.2aspects.useful-credible $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "2aspects.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

    $trec_eval -q -c -M 1000 -m cam_map_three -R qrels_threeaspects $QRELS/misinfo-qrels.3aspects $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "3aspects" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

    $trec_eval -q -c -M 1000 -m ndcg -R qrels $QRELS/misinfo-qrels-binary.useful $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "binary.useful" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
    $trec_eval -q -c -M 1000 -m ndcg -R qrels $QRELS/misinfo-qrels-binary.useful-correct $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "binary.useful-correct" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
    $trec_eval -q -c -M 1000 -m ndcg -R qrels $QRELS/misinfo-qrels-binary.useful-credible $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "binary.useful-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
    $trec_eval -q -c -M 1000 -m ndcg -R qrels $QRELS/misinfo-qrels-binary.useful-correct-credible $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "binary.useful-correct-credible" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY

    python3 $compatibility $QRELS/misinfo-qrels-graded.harmful-only $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "graded.harmful-only" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
    python3 $compatibility $QRELS/misinfo-qrels-graded.helpful-only $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "graded.helpful-only" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
done
    
# Recall runs

for RUN_FULL_PATH in $RECALL/*
do
    RUN_NAME=`basename $RUN_FULL_PATH`
    SUMMARY="$RECALL_OUT_DIR/$RUN_NAME.summary"
    printf "run\tqrels\tmeasure\ttopic\tscore\n" > $SUMMARY
    $trec_eval -q -c -M 10000 -m Rprec -R qrels $QRELS/misinfo-qrels-binary.incorrect $RUN_FULL_PATH | gawk '{print "'$RUN_NAME'" "\t" "binary.incorrect" "\t" $1 "\t" $2 "\t" $3}' >> $SUMMARY
done





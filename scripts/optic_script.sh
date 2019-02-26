#!/bin/zsh
. ./paths.sh


for f in ~/bayes/other_code/confounder_detection_linear/data/optical_device/*.txt; do
    seed=$(pwgen 32 1)
    export THEANO_FLAGS="base_compiledir=$THEANOPATH/theano_$seed"
    cd $RUNPATH
    $PYTHONPATH run_on_any_data.py $f "test.csv" &
done

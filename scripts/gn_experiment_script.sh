#!/bin/zsh
. ./paths.sh

rm -rf "$THEANOPATH/theano_*"
for b in {1..50}
do
    for n in {5..14}
    do
        seed=$(pwgen 32 1)
        export THEANO_FLAGS="base_compiledir=$THEANOPATH/theano_$seed"
        cd $RUNPATH
        $PYTHONPATH gn_experiments.py -n $n -b $b -r 2 &
    done
    wait
done

#!/bin/zsh
. ./paths.sh
name=$(uname -n) 


N=500                  # The number of for a single X
DX=(2 4 6 8 10)      # Dimensions of X
DZ=(2 4 6 8 10)      # Dimensions of the true Z
L=("n" "lap" "u" "ln") # Distributions to use: Normal, Laplace, Uniform, Lognormal
batch=10               # The size of the batch to be computed at once. See below what this does and why we need it

rm "$RUNPATH/nohup_$name.out"
rm -rf "$THEANOPATH/theano_*"

# Keeping pymc/theano alive while running it over a large number of datasets (in parallel) leads to exploding
# memory usage and python MemoryError. To avoid this, the code internally has a 'batch size' and we restart the
# processes to keep memory usage low. Even if a MemoryError occurs, the new process will still pick up where
# we left off.
for i in {1..5}
do
    for label in "${L[@]}"
    do
        for dz in "${DZ[@]}"
        do
            for dx in "${DX[@]}"
            do
                # To run theano for multiple datasets in parallel, each process requires its own compiledir
                seed=$(pwgen 32 1)
                export THEANO_FLAGS="base_compiledir=$THEANOPATH/theano_$seed"
                cd $RUNPATH
                nohup $PYTHONPATH synthetic_experiments.py -n $N -dx $dx -dz $dz -label $label -data "heat" -batch $batch >> "nohup_$name.out" 2>&1 &
            done
            wait
        done
    done
done

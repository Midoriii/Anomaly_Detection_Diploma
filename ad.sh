#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=12gb:scratch_local=1gb:ngpus=2
#PBS -l walltime=1:00:00 

DATADIR=/storage/brno2/home/apprehension

module add python-3.6.2-gcc
module add python36-modules-gcc


cp -R $DATADIR/autoencoder_tester.py $DATADIR/Graphs $DATADIR/Models $DATADIR/Data $DATADIR/Reconstructed $SCRATCHDIR


cd $SCRATCHDIR


python autoencoder_tester.py


cp -R Graphs Reconstructed $DATADIR

clean_scratch
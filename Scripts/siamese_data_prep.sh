#!/bin/bash
#PBS -l select=1:ncpus=1:mem=40gb:scratch_local=40gb
#PBS -l walltime=1:00:00

cd ..
cd ..

DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc


cp -R $DATADIR/siamese_network_data_prep.py $DATADIR/Data $SCRATCHDIR


cd $SCRATCHDIR


python siamese_network_data_prep.py


cp -vr $SCRATCHDIR/Data/* $DATADIR/Data/

clean_scratch

#!/bin/bash
#PBS -l select=1:ncpus=1:mem=300gb:scratch_local=60gb
#PBS -l walltime=3:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc


cp -R $DATADIR/siamese_network_data_prep.py $DATADIR/Data $SCRATCHDIR


cd $SCRATCHDIR

mkdir -p DataHuge


python siamese_network_data_prep.py


cp -vr $SCRATCHDIR/DataHuge/* $DATADIR/DataHuge/

clean_scratch

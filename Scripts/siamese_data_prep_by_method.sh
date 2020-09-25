#!/bin/bash
#PBS -l select=1:ncpus=1:mem=40gb:scratch_local=60gb
#PBS -l walltime=3:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add opencv-3.4.5-py36


cp -R $DATADIR/siamese_network_data_prep_by_method.py $DATADIR/reshape_util.py $DATADIR/Clonky-ok $DATADIR/Clonky-vadne $SCRATCHDIR

cd $SCRATCHDIR

mkdir -p DataHuge
mkdir -p Data


python siamese_network_data_prep_by_method.py


cp -vr $SCRATCHDIR/DataHuge/* $DATADIR/DataHuge/
cp -vr $SCRATCHDIR/Data/* $DATADIR/Data/

clean_scratch

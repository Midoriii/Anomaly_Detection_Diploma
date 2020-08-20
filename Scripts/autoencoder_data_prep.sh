#!/bin/bash
#PBS -l select=1:ncpus=1:mem=15gb:scratch_local=10gb
#PBS -l walltime=1:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add opencv-3.4.5-py36


cp -R $DATADIR/autoencoder_data_prep.py $DATADIR/reshape_util.py $DATADIR/Clonky-ok $DATADIR/Clonky-ok-filtered $DATADIR/Clonky-vadne $DATADIR/Clonky-vadne-full $SCRATCHDIR

cd $SCRATCHDIR

mkdir -p Data


python autoencoder_data_prep.py


cp -vr $SCRATCHDIR/Data/* $DATADIR/Data/

clean_scratch

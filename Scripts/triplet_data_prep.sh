#!/bin/bash
#PBS -l select=1:ncpus=1:mem=40gb:scratch_local=60gb
#PBS -l walltime=3:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add opencv-3.4.5-py36


cp -R $DATADIR/triplet_data_prep.py $DATADIR/Data $SCRATCHDIR

cd $SCRATCHDIR

mkdir -p DataTriplet


python triplet_data_prep.py


cp -vr $SCRATCHDIR/DataTriplet/* $DATADIR/DataTriplet/

clean_scratch

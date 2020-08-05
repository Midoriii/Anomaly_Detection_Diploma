#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:ngpus=1:mem=150gb:scratch_local=60gb
#PBS -l walltime=1:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10


cp -R $DATADIR/siamese_predict_tester_se.py $DATADIR/DataHuge $DATADIR/Model_Saves $SCRATCHDIR

cd $SCRATCHDIR

python siamese_predict_tester_se.py

clean_scratch

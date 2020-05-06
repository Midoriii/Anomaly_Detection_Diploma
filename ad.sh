#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=6gb:ngpus=1:scratch_local=1gb
#PBS -l walltime=10:00:00 

DATADIR=/storage/brno2/home/apprehension

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10


cp -R $DATADIR/autoencoder_tester.py $DATADIR/Models $DATADIR/Data $SCRATCHDIR


cd $SCRATCHDIR
mkdir Graphs
mkdir Reconstructed


python autoencoder_tester.py


cp -vr $SCRATCHDIR/Graphs/* $DATADIR/Graphs/
cp -vr $SCRATCHDIR/Reconstructed/* $DATADIR/Reconstructed/

clean_scratch
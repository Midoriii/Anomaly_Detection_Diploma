#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=15gb:ngpus=1:scratch_local=3gb
#PBS -l walltime=1:30:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10


cp -R $DATADIR/bigan_tester.py $DATADIR/Models $DATADIR/Data $SCRATCHDIR


cd $SCRATCHDIR
mkdir -p Graphs/{Losses,biGANErrors}
mkdir -p Model_Saves/{Detailed,Weights}


python bigan_tester.py -e 5000 -b 16 -m BasicBiganXEntropy -t BSE


cp -vr $SCRATCHDIR/Graphs/Losses/* $DATADIR/Graphs/Losses/
cp -vr $SCRATCHDIR/Graphs/biGANErrors/* $DATADIR/Graphs/biGANErrors/
cp -vr $SCRATCHDIR/Model_Saves/Detailed/* $DATADIR/Model_Saves/Detailed/
cp -vr $SCRATCHDIR/Model_Saves/Weights/* $DATADIR/Model_Saves/Weights/

clean_scratch
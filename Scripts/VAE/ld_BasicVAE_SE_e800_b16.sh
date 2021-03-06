#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=8gb:ngpus=1:scratch_local=3gb:cluster=adan
#PBS -l walltime=3:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10


cp -R $DATADIR/vae_tester.py $DATADIR/Models $DATADIR/Data $SCRATCHDIR


cd $SCRATCHDIR
mkdir -p Graphs/{Losses,VAEScores,VAEReco}
mkdir -p Model_Saves/{Detailed,Weights}


python vae_tester.py -e 800 -b 16 -m BasicVAE -t SE


cp -vr $SCRATCHDIR/Graphs/Losses/* $DATADIR/Graphs/Losses/
cp -vr $SCRATCHDIR/Graphs/VAEScores/* $DATADIR/Graphs/VAEScores/
cp -vr $SCRATCHDIR/Graphs/VAEReco/* $DATADIR/Graphs/VAEReco/
cp -vr $SCRATCHDIR/Model_Saves/Detailed/* $DATADIR/Model_Saves/Detailed/
cp -vr $SCRATCHDIR/Model_Saves/Weights/* $DATADIR/Model_Saves/Weights/

clean_scratch

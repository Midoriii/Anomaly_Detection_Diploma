#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=12gb:ngpus=1:scratch_local=3gb
#PBS -l walltime=5:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10


cp -R $DATADIR/autoencoder_tester.py $DATADIR/Models $DATADIR/Data $SCRATCHDIR


cd $SCRATCHDIR
mkdir -p Graphs/{Losses,ReconstructionErrors}
mkdir -p Reconstructed/Error_Arrays
mkdir -p Model_Saves/{Detailed,Weights}


python autoencoder_tester.py -e 600 -b 4 -m TransposeConvAutoencoderDeepExtraLLR


cp -vr $SCRATCHDIR/Graphs/Losses/* $DATADIR/Graphs/Losses/
cp -vr $SCRATCHDIR/Graphs/ReconstructionErrors/* $DATADIR/Graphs/ReconstructionErrors/
cp -vr $SCRATCHDIR/Reconstructed/* $DATADIR/Reconstructed/
cp -vr $SCRATCHDIR/Reconstructed/Error_Arrays/* $DATADIR/Reconstructed/Error_Arrays/
cp -vr $SCRATCHDIR/Model_Saves/Detailed/* $DATADIR/Model_Saves/Detailed/
cp -vr $SCRATCHDIR/Model_Saves/Weights/* $DATADIR/Model_Saves/Weights/

clean_scratch

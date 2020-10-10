#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=70gb:ngpus=1:scratch_local=80gb
#PBS -l walltime=12:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10


cp -R $DATADIR/triplet_network_tester.py $DATADIR/Models $DATADIR/Data $DATADIR/DataTriplet $SCRATCHDIR


cd $SCRATCHDIR
mkdir -p Graphs/{Losses,TripletScores}
mkdir -p Model_Saves/{Detailed,Weights}


python triplet_network_tester.py -e 60 -m BasicTripletNet -t SE -s 3

cp -vr $SCRATCHDIR/Graphs/Losses/* $DATADIR/Graphs/Losses/
cp -vr $SCRATCHDIR/Graphs/TripletScores/* $DATADIR/Graphs/TripletScores/
cp -vr $SCRATCHDIR/Model_Saves/Detailed/* $DATADIR/Model_Saves/Detailed/
cp -vr $SCRATCHDIR/Model_Saves/Weights/* $DATADIR/Model_Saves/Weights/

clean_scratch

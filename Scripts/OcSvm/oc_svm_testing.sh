#!/bin/bash
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=16gb:ngpus=1:scratch_local=10gb
#PBS -l walltime=5:00:00


DATADIR=/storage/brno6/home/apprehension

cd $DATADIR

module add python-3.6.2-gcc
module add python36-modules-gcc
module add tensorflow-1.13.1-gpu-python3
module add opencv-3.4.5-py36
module add cuda-10.0
module add cudnn-7.4.2-cuda10

cd $SCRATCHDIR

mkdir -p Model_Saves/Detailed/OcSvm

cp -R $DATADIR/Model_Saves/Detailed/OcSvm/* $SCRATCHDIR/Model_Saves/Detailed/OcSvm
cp -R $DATADIR/oc_svm_tester.py $DATADIR/Data $SCRATCHDIR


python oc_svm_tester.py


clean_scratch

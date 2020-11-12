#!/bin/bash
FILES=*.sh
for f in $FILES
do
  qsub $f
done
#!/bin/bash
FILES=ld_*.sh
for f in $FILES
do
  qsub $f
done
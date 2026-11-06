#!/bin/bash
# imports every archive on the drive into ~/feagent-mh/benchmarks/gold, one gold.py process per archive
cd ~/feagent-mh
A=/media/dell/jaimesThumbdrive/gold_benchmarks/archives
R=~/gold-catalog/import
: > $R/results.tsv
for f in $(ls $A/*.tgz | sort); do
  n=$(basename $f .tgz)
  t0=$(date +%s.%N)
  ~/gold-catalog/scripts/wd.sh imp-$n $R/$n.log PYTHONPATH=$HOME/feagent-mh/meshing-harness/src:$HOME/feagent-mh -- scripts/gold.py --root $HOME/feagent-mh/benchmarks/gold import $f
  rc=$?
  t1=$(date +%s.%N)
  printf "%s\t%s\t%.1f\n" $n $rc $(echo "$t1 - $t0" | bc) >> $R/results.tsv
done
echo DONE >> $R/results.tsv

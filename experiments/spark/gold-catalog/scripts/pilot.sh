#!/bin/bash
cd ~/gold-catalog
E="PYTHONPATH=$HOME/feagent-mh/meshing-harness/src:$HOME/feagent-mh"
for s in abc_00000627_bc4 abc_00000974_bc4 abc_00001748_bc0 abc_00000330_bc1 abc_00000223_bc8; do
  p=${s%_bc*}; W=pilot/$p
  mkdir -p $W
  t0=$(date +%s.%N); scripts/wd.sh pm-$s $W/mesh.out $E -- scripts/gold_case.py mesh $p $W --sets $s; r1=$?; t1=$(date +%s.%N)
  scripts/wd.sh pc-$s $W/case.out $E -- scripts/gold_case.py case $s $W; r2=$?; t2=$(date +%s.%N)
  scripts/wd.sh ps-$s $W/solve.out $E -- scripts/gold_case.py solve $s $W; r3=$?; t3=$(date +%s.%N)
  echo -e "$s\t$r1\t$r2\t$r3\t$(echo "$t1-$t0" | bc)\t$(echo "$t2-$t1" | bc)\t$(echo "$t3-$t2" | bc)" >> pilot/pilot.tsv
done
echo DONE >> pilot/pilot.tsv

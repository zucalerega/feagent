#!/bin/bash
# usage: run_one.sh h method   -> appends RESULT to ~/bench/raw.log, watchdog log ~/bench/watchdog.log
h=$1; m=$2; U=bench-$h-$m-$$
cd ~/feagent
systemd-run --user --scope -q --unit=$U -p MemoryMax=20G ~/micromamba/envs/feagent/bin/python ~/bench/bench_spark.py --h $h --method $m > ~/bench/out_${h}_$m.txt 2>&1 &
JP=$!; min=999999999; killed=0
while kill -0 $JP 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ $a -lt $min ] && min=$a
  if [ $a -lt 10485760 ]; then systemctl --user stop $U.scope; killed=1; echo "$(date +%T) KILLED $h $m avail_kB=$a" >> ~/bench/watchdog.log; fi
  sleep 1
done
wait $JP; rc=$?
echo "$(date +%T) h=$h method=$m rc=$rc killed=$killed min_MemAvailable_GiB=$(awk -v x=$min 'BEGIN{printf "%.2f",x/1048576}')" >> ~/bench/watchdog.log
grep RESULT ~/bench/out_${h}_$m.txt >> ~/bench/raw.log
tail -3 ~/bench/out_${h}_$m.txt | cut -c1-1500

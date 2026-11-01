#!/bin/bash
# usage: wd.sh NAME LOGFILE [ENV=VAL ...] -- python-args...
# runs ~/micromamba/envs/feagent/bin/python under a 40G user scope; kills it if MemAvailable < 12 GiB
name=$1; log=$2; shift 2
envs=()
while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
U=gold-$name-$$
systemd-run --user --scope -q --unit=$U -p MemoryMax=40G env "${envs[@]}" ~/micromamba/envs/feagent/bin/python "$@" > "$log" 2>&1 &
JP=$!; min=999999999; killed=0
while kill -0 $JP 2>/dev/null; do
  a=$(awk "/MemAvailable/{print \$2}" /proc/meminfo)
  [ $a -lt $min ] && min=$a
  if [ $a -lt 12582912 ]; then systemctl --user stop $U.scope; killed=1; echo "$(date +%T) KILLED $name avail_kB=$a" >> ~/gold-catalog/logs/watchdog.log; fi
  sleep 1
done
wait $JP; rc=$?
echo "$(date +%T) $name rc=$rc killed=$killed min_MemAvailable_GiB=$(awk -v x=$min "BEGIN{printf \"%.2f\",x/1048576}")" >> ~/gold-catalog/logs/watchdog.log
exit $rc

#!/bin/bash
# usage: run_case.sh <case_dir>  -- cuDSS solve under a 60G user scope + 1 s MemAvailable watchdog (12 GiB floor)
d=$(realpath "$1"); name=$(basename "$d"); U=wingbox-$name-$$
LOG=~/runs/wingbox-2026-10-31/watchdog.log
cd ~/feagent-mh || exit 9
export PYTHONPATH=$HOME/feagent-mh
base=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
t0=$(date +%s.%N)
systemd-run --user --scope -q --unit=$U -p MemoryMax=60G ~/micromamba/envs/feagent/bin/python -m feacore.cli \
  "$d/case.json" --out "$d/result.json" --vtu "$d/result.vtu" --linear-solver cudss > "$d/solve.log" 2>&1 &
JP=$!; min=999999999; killed=0; peak=0
CG=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/app.slice/$U.scope
while kill -0 $JP 2>/dev/null; do
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$a" -lt "$min" ] && min=$a
  p=$(cat $CG/memory.peak 2>/dev/null || echo 0); [ "$p" -gt "$peak" ] 2>/dev/null && peak=$p
  if [ "$a" -lt 12582912 ]; then systemctl --user stop $U.scope; killed=1; echo "$(date +%T) KILLED $name avail_kB=$a" >> $LOG; fi
  sleep 1
done
wait $JP; rc=$?
t1=$(date +%s.%N)
line="$(date +%T) case=$name rc=$rc killed=$killed wall_s=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f",b-a}') base_MemAvailable_GiB=$(awk -v x=$base 'BEGIN{printf "%.2f",x/1048576}') min_MemAvailable_GiB=$(awk -v x=$min 'BEGIN{printf "%.2f",x/1048576}') drop_GiB=$(awk -v x=$base -v y=$min 'BEGIN{printf "%.2f",(x-y)/1048576}') cgroup_peak_GiB=$(awk -v x=$peak 'BEGIN{printf "%.2f",x/1073741824}')"
echo "$line" >> $LOG; echo "$line" > "$d/watchdog.txt"; echo "$line"
tail -5 "$d/solve.log" | cut -c1-600

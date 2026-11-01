#!/bin/bash
# 3x2 grid of the six systems' crush videos for one part, same layout as the linear all-systems grid.
# usage: crush_grid.sh PART SUFFIX(crush|crush_rtx) VIDEO_DIR OUT.mp4
part=$1; suf=$2; V=$3; out=$4
systems=(harness_auto harness_expert ace_sizing gold_nested netgen harness_hex)
inputs=(); filt=""; i=0
for s in "${systems[@]}"; do
  f=$V/${s}_${part}_${suf}.mp4
  if [ -f "$f" ]; then
    inputs+=(-i "$f")
    filt+="[$i:v]scale=1280:720,setsar=1,fps=30,tpad=stop_mode=clone:stop_duration=30[v$i];"
  else
    inputs+=(-f lavfi -i "color=c=0x1d2330:s=1280x720:r=30:d=16")
    filt+="[$i:v]drawtext=text='${s//_/ }  -  crush still running':fontcolor=white:fontsize=40:x=(w-tw)/2:y=(h-th)/2,setsar=1[v$i];"
  fi
  i=$((i+1))
done
filt+="[v0][v1][v2][v3][v4][v5]xstack=inputs=6:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0,trim=duration=16[out]"
ffmpeg -loglevel error -y "${inputs[@]}" -filter_complex "$filt" -map "[out]" -c:v libx264 -pix_fmt yuv420p -crf 20 -t 16 "$out"

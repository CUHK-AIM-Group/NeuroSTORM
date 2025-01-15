#!/bin/sh

for ((i=1; i<22; i++)); do 
  v=`sed -ne "${i}p" labels` 
  avwmaths++ $1 -thr $i -uthr $i -bin -mul $v temp-$i
done

imcp temp-1 $1
for ((i=2; i<22; i++)); do 
  avwmaths++ $1 -add temp-$i $1
done

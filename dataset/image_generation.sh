#!/bin/sh
max=6000

mkdir dataset/generated_videos/$1

for num in `seq 0 $max`
do
    rm -rf ./tmp/*.json
    rm -rf ./tmp/img/*.png

    echo "$num"
    timeout --kill-after=100 100 python3 recreation_v3.py --jsonold dataset/annotations/$1/example_$num.json
    if [ $? -eq 124 ]; then
        num=num-1
    else
        ffmpeg -framerate 25 -i ./dist/img_%04d.png -c:v copy dataset/generated_videos/$1/example_$num.mkv
    fi
done
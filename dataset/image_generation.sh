#!/bin/sh

unzip annotations.zip

function unzip_file()
{
    for file in `ls $1`
    do
        cur=$1/$file
        if [ -d $cur ]; then
            unzip_file $cur
        else
            unzip $cur -d $1
        fi
    done
}

unzip_file "icqa_dataset"

function process_file()
{
    for file in `ls $1`
    do
        cur=$1/$file
        if [ -d $cur ]; then
            process_file $cur
        else
            if [[ "$file" == *".json" ]]; then 
                echo $cur
                timeout --kill-after=100 100 python recreate.py --jsonold $cur
                if [ $? -eq 124 ]; then
                    echo "Timeout out"
                else
                    mkdir -p generated_videos/$1
                    ffmpeg -framerate 25 -i ./dist/img_%04d.png -c:v copy generated_videos/$1/${file:0:-5}.mkv
                fi
            fi
        fi
    done
}

process_file "icqa_dataset"

#!/bin/bash

while getopts m:c:i:t:o:f: flag
do
    case "${flag}" in
        m) mozzapath=${OPTARG};;
		c) containerpath=${OPTARG};;
        i) input=${OPTARG};;
        t) model=${OPTARG};;
        o) output=${OPTARG};;
    esac
done

echo "input: $input";
echo "output: $output";
echo "mozzapath: $mozzapath";
echo "containerpath: $containerpath";
echo "model: $model";

docker run -v $containerpath:/data -v $mozzapath:/mozza-path creamlab/mozza mozza-templater -b -m /mozza-path/data/in/shape_predictor_68_face_landmarks.dat -n /data/$input -s /data/$model -o /data/$output
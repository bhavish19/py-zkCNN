./#!/bin/bash

set -x

./build.sh
/usr/bin/cmake --build ../cmake-build-release --target demo_alexnet_run -- -j 6

run_file=../cmake-build-release/src/demo_alexnet_run
out_file=../output/single/demo-result-alexnet.txt

mkdir -p ../output/single
mkdir -p ../log/single

alexnet_i=../data/alexnet.cifar10.relu.max/alexnet.cifar.relu-1-images-weights-qint8.csv
alexnet_c=../data/alexnet.cifar10.relu.max/alexnet.cifar.relu-1-scale-zeropoint-uint8.csv
alexnet_o=../output/single/alexnet.cifar.relu-1-infer.csv

${run_file} ${alexnet_i} ${alexnet_c} ${alexnet_o} 1 > ${out_file}
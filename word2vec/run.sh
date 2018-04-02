#! /bin/bash


set -x

make clean;make

time ./bin/w2v_bin -train text8 -output model1 -cbow 1 -size 32 -window 8 -negative 5 -hs 0 -sample 1e-4 -threads 1 -binary 0 -iter 1




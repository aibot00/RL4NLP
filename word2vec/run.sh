#! /bin/bash


set -x

make


time ./bin/word2vec -train data/questions-words.txt -output model1 -cbow 1 -size 32 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 1

time ./bin/w2v_bin -train data/questions-words.txt -output model2 -cbow 1 -size 32 -window 5 -negative 5 -hs 0 -sample 1e-4 -threads 1 -binary 1 -iter 1



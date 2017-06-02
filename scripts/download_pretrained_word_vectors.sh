#!/bin/bash

THIS_DIR=$(dirname $(readlink -f "$0"))
DATASET_ROOT_DIR="$THIS_DIR/../var/pretrained-vectors"
mkdir -p $DATASET_ROOT_DIR
cd $DATASET_ROOT_DIR

echo 'Downloading and extracting GloVe..'
mkdir -p 'glove'
cd 'glove'
wget -nc http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip -n glove.42B.300d.zip > /dev/null && rm glove.42B.300d.zip
cd ..

echo 'Downloading and extracting fasttext..'
mkdir -p 'fasttext'
cd 'fasttext'
wget -nc https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.en.vec
cd ..

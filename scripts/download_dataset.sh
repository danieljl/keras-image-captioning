#!/bin/bash

THIS_DIR=$(dirname $(readlink -f "$0"))
cd "$THIS_DIR/../var/datasets"

echo 'Downloading and extracting Flickr8k dataset...'
mkdir -p flickr8k
cd flickr8k
wget -nc http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip
wget -nc http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
unzip -n Flickr8k_text.zip -x '__MACOSX/*' -d Flickr8k_text > /dev/null
unzip -nj Flickr8k_Dataset.zip -x '__MACOSX/*' -d Flickr8k_Dataset > /dev/null
echo 'Done!'

#!/bin/bash

THIS_DIR=$(dirname $(readlink -f "$0"))
PYCOCOEVALCAP_DIR="$THIS_DIR/../pycocoevalcap"

git clone https://github.com/tylin/coco-caption.git /tmp/coco-caption
cd /tmp/coco-caption/pycocoevalcap/meteor
cp -r meteor-1.5.jar data "$PYCOCOEVALCAP_DIR/meteor"

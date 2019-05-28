#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "USAGE: $0 DATA_DIR"
    exit 1
fi

DATA_DIR=$(realpath $1)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo "DATA_DIR=$DATA_DIR; SCRIPT_DIR=$SCRIPT_DIR"

mkdir -pv $DATA_DIR

TRAIN_0=train_0
TRAIN_1=train_1
TRAIN_2=train_2
VAL=validation

# Download ----------
DOWNLOAD_DIR=$DATA_DIR/download
mkdir -p $DOWNLOAD_DIR
pushd $DOWNLOAD_DIR
for DIR in $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL; do
    TAR=${DIR}.tar.gz
    if [ ! -f "$TAR" ]; then
        aws s3 --no-sign-request cp s3://open-images-dataset/tar/$TAR $TAR
    else
        echo "Found $TAR..."
    fi
done

progress () {
    COUNTER=0
    while read line; do
        COUNTER=$((COUNTER+1))
        echo -ne "\rExtracting $line; Unpacked $COUNTER files."
    done
}

for DIR in $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL; do
    if [ -d $DIR ]; then
        continue
    fi
    TAR=${DIR}.tar.gz
    if [ ! -f $TAR ]; then
        echo "ERROR: Expected $TAR in $DOWNLOAD_DIR"
        exit 1
    fi
    echo "Unpacking $TAR..."
    tar xvf $TAR | progress
done
popd

# Convert ----------
OUT_DIR=$DATA_DIR/imported
pushd $SCRIPT_DIR
echo "Converting, saving in $OUT_DIR..."
python import_train_images.py $DOWNLOAD_DIR $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL --out_dir $OUT_DIR

# Move to 1 folder ----------
FINAL_TRAIN_DIR=$DATA_DIR/train_oi
FINAL_VAL_DIR=$DATA_DIR/val_oi
mkdir -p $FINAL_TRAIN_DIR
mkdir -p $FINAL_VAL_DIR

for CLEAN_DIR in $OUT_DIR/*_clean; do
    echo "mv $CLEAN_DIR/* $FINAL_TRAIN_DIR"
    mv $CLEAN_DIR/* $FINAL_TRAIN_DIR
done

# Update Cache ----------
CACHE_P=$DATA_DIR/cache.pkl
export PYTHONPATH=$(pwd)

echo "Updating cache $CACHE_P..."
python dataloaders/images_loader.py update $FINAL_TRAIN_DIR "$CACHE_P" --min_size 128
python dataloaders/images_loader.py update $FINAL_VAL_DIR "$CACHE_P" --min_size 128

echo "----------------------------------------"
echo "Done"
echo "To train, you MUST UPDATE configs/dl/oi.cf:"
echo ""
echo "  image_cache_pkl = '$1/cache.pkl'"
echo "  train_imgs_glob = '$1/train_oi'"
echo "  val_glob = '$1/val_oi'"
echo ""
echo "----------------------------------------"

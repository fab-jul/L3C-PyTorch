#!/bin/bash

set -e

if [[ -z $1 ]]; then
    echo "USAGE: $0 DATA_DIR [OUT_DIR]"
    exit 1
fi

DATA_DIR=$(realpath $1)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

if [[ -n $2 ]]; then
  OUT_DIR=$2
else
  OUT_DIR=$DATA_DIR
fi

progress () {
    COUNTER=0
    while read LINE; do
        COUNTER=$((COUNTER+1))
        if [[ $((COUNTER % 10)) == 0 ]]; then
            echo -ne "\rExtracting $LINE; Unpacked $COUNTER files."
        fi
    done
    echo ""
}

echo "DATA_DIR=$DATA_DIR;\nSCRIPT_DIR=$SCRIPT_DIR;\nSaving to $OUT_DIR"

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
        echo "Downloading $TAR..."
        aws s3 --no-sign-request cp s3://open-images-dataset/tar/$TAR $TAR
    else
        echo "Found $TAR..."
    fi
done

for DIR in $TRAIN_0 $TRAIN_1 $TRAIN_2 $VAL; do
    TAR=${DIR}.tar.gz
    if [ -d $DIR ]; then
        echo "Found $DIR, not unpacking $TAR..."
        continue
    fi
    if [ ! -f $TAR ]; then
        echo "ERROR: Expected $TAR in $DOWNLOAD_DIR"
        exit 1
    fi
    echo "Unpacking $TAR..."
    ( tar xvf $TAR | progress ) &
done

# Wait for all unpacking background processes
wait
echo "Unpacked all!"

popd

# Convert ----------
FINAL_TRAIN_DIR=$DATA_DIR/train_oi
FINAL_VAL_DIR=$DATA_DIR/val_oi

DISCARD=$OUT_DIR/discard
DISCARD_VAL=$OUT_DIR/discard_val
pushd $SCRIPT_DIR

echo "Importing train..."
# NOTE: this is were you want to employ parallelization on a cluster if it's available.
# See import_train_images.py
python import_train_images.py $DOWNLOAD_DIR $TRAIN_0 $TRAIN_1 $TRAIN_2 \
       --out_dir_clean=$FINAL_TRAIN_DIR \
       --out_dir_discard=$DISCARD \
       --resolution=512

python import_train_images.py $DOWNLOAD_DIR $VAL \
       --out_dir_clean=$FINAL_VAL_DIR \
       --out_dir_discard=$DISCARD_VAL \
       --resolution=512

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
echo "  train_imgs_glob = '$(realpath $1/train_oi)'"
echo "  val_glob = '$(realpath $1/val_oi)'"
echo ""
echo "----------------------------------------"

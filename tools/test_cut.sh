#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh
source "${BASH_SOURCE%/*}"/install_dependencies.sh

NETWORK_NAME=$1
CONSTANT_TEST_ARGUMENTS="--num_test 250" # --checkpoints_dir /tmp/borstelmanna0/remote-checkpoints"
DATASET=serengeti-incandescent

for EPOCH in "${@:2}"
do
	conda run --no-capture-output --live-stream -n contrastive-unpaired-translation --cwd $CUT_DIR python test.py --dataroot "$CYCLE_GAN_DIR/datasets/$DATASET/" --name "$NETWORK_NAME" $CONSTANT_TEST_ARGUMENTS --epoch "$EPOCH" --results_dir /tmp/borstelmanna0/results;
	conda run --no-capture-output --live-stream -n contrastive-unpaired-translation --cwd $CUT_DIR python test.py --dataroot "$CYCLE_GAN_DIR/datasets/$DATASET/" --name "$NETWORK_NAME" $CONSTANT_TEST_ARGUMENTS --epoch "$EPOCH" --results_dir /tmp/borstelmanna0/results-large --preprocess scale_width --load_size 1024;
done

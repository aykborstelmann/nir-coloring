#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh
source "${BASH_SOURCE%/*}"/install_dependencies.sh

NETWORK_NAME=$1
CONSTANT_NETWORK_ARGUMENTS="--no_dropout --netG unet_256 --norm instance"

rm -r $CYCLE_GAN_DIR/results

for EPOCH in "${@:2}"
do
	cp "$CYCLE_GAN_DIR/checkpoints/$NETWORK_NAME/${EPOCH}_net_G_A.pth" "$CYCLE_GAN_DIR/checkpoints/$NETWORK_NAME/${EPOCH}_net_G.pth"
	conda run --no-capture-output --live-stream -p "$CYCLE_GAN_ENV_PREFIX" --cwd $CYCLE_GAN_DIR python test.py --dataroot ./datasets/caltech/testA/ --name "$NETWORK_NAME" --model test $CONSTANT_NETWORK_ARGUMENTS --epoch "$EPOCH";
	conda run --no-capture-output --live-stream -p "$CYCLE_GAN_ENV_PREFIX" --cwd $CYCLE_GAN_DIR python test.py --dataroot ./datasets/caltech/testA/ --name "$NETWORK_NAME" --model test $CONSTANT_NETWORK_ARGUMENTS --epoch "$EPOCH" --results_dir ./results-large --preprocess scale_width --load_size 1024;
done
#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh
source "${BASH_SOURCE%/*}"/install_dependencies.sh

NETWORK_NAME=$1
CONSTANT_NETWORK_ARGUMENTS="--no_dropout --netG unet_256 --norm instance"
CONSTANT_TEST_ARGUMENTS="--num_test 250" # --checkpoints_dir /tmp/borstelmanna0/remote-checkpoints"
DATASET=serengeti-incandescent

rm -r $CYCLE_GAN_DIR/results

for EPOCH in "${@:2}"
do
	cp "$CYCLE_GAN_DIR/checkpoints/$NETWORK_NAME/${EPOCH}_net_G_A.pth" "$CYCLE_GAN_DIR/checkpoints/$NETWORK_NAME/${EPOCH}_net_G.pth"
	conda run --no-capture-output --live-stream -n cycleGAN --cwd $CYCLE_GAN_DIR python test.py --dataroot "./datasets/$DATASET/testA/" --name "$NETWORK_NAME" --model test $CONSTANT_NETWORK_ARGUMENTS $CONSTANT_TEST_ARGUMENTS --epoch "$EPOCH" --results_dir /tmp/borstelmanna0/results;
	conda run --no-capture-output --live-stream -n cycleGAN --cwd $CYCLE_GAN_DIR python test.py --dataroot "./datasets/$DATASET/testA/" --name "$NETWORK_NAME" --model test $CONSTANT_NETWORK_ARGUMENTS $CONSTANT_TEST_ARGUMENTS --epoch "$EPOCH" --results_dir /tmp/borstelmanna0/results-large --preprocess scale_width --load_size 1024;
	rm "$CYCLE_GAN_DIR/checkpoints/$NETWORK_NAME/${EPOCH}_net_G.pth"
done

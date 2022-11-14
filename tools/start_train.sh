#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh
source "${BASH_SOURCE%/*}"/install_dependencies.sh

NETWORK_NAME=$1
CONSTANT_NETWORK_ARGUMENTS="--save_epoch_freq=40 --netG unet_256 --norm instance --verbose --use_wandb --wandb_project_name cycle-gan"

if [ ! $NETWORK_NAME ]; then
  echo "Usage: start_train <network-name>"
  exit
fi;

conda run --no-capture-output --live-stream -p "$CYCLE_GAN_ENV_PREFIX" --cwd $CYCLE_GAN_DIR python train.py --dataroot ./datasets/caltech --name $NETWORK_NAME --model cycle_gan $CONSTANT_NETWORK_ARGUMENTS >> $NETWORK_NAME.out 2>&1 &
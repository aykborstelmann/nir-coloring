#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh

if [ ! "$DEFAULT_REMOTE" ]; then
  echo "Missing environment variable \"DEFAULT_REMOTE\""
  exit
fi

ARGUMENTS="${*:2}"
ssh "$DEFAULT_REMOTE" "source ~/.bashrc; $REMOTE_REPO_PATH/tools/$1.sh $ARGUMENTS"

if [ "$1" = "test" ]; then
  echo "Syncing files"

  rsync -v -r "$DEFAULT_REMOTE":"$REMOTE_REPO_PATH"/cycle-gan/results/ "$CYCLE_GAN_DIR"/results
  rsync -v -r "$DEFAULT_REMOTE":"$REMOTE_REPO_PATH"/cycle-gan/results-large/ "$CYCLE_GAN_DIR"/results-large

  find "$CYCLE_GAN_DIR"/results/"$2"/test_*/index.html | sort | xargs -n 1 xdg-open
  find "$CYCLE_GAN_DIR"/results-large/"$2"/test_*/index.html | sort | xargs -n 1 xdg-open

  rsync -v -r "$DEFAULT_REMOTE":"$REMOTE_REPO_PATH"/cycle-gan/checkpoints/ "$CYCLE_GAN_DIR"/checkpoints
fi

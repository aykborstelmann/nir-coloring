#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh

if [ ! $SESSION_TYPE = "remote/ssh" ]; then
    echo "You are currently in a local session";
    echo "Cleanup should only be executed on a remote system";
    exit 1
fi;

rm -r "$CYCLE_GAN_DIR"/checkpoints
rm -r "$CYCLE_GAN_DIR"/results
rm -r "$CYCLE_GAN_DIR"/results-large
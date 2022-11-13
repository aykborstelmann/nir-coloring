#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

install_dependencies() {
  if ! find_in_conda_env NIR_COLORING_ENV_PREFIX; then
    XDG_CACHE_HOME=/tmp/.cache/ conda env create --prefix $NIR_COLORING_ENV_PREFIX --file "$ROOT_DIR/environment.yml";
    conda develop -p $NIR_COLORING_ENV_PREFIX nir-coloring
  fi;

  if ! find_in_conda_env $CYCLE_GAN_ENV_PREFIX; then
    XDG_CACHE_HOME=/tmp/.cache/ conda env create --prefix $CYCLE_GAN_ENV_PREFIX --file "$ROOT_DIR/cycle-gan/environment.yml";
  fi;
}

install_dependencies
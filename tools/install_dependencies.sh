#!/usr/bin/env bash

source "${BASH_SOURCE%/*}"/common.sh

find_in_conda_env(){
    conda env list | grep "${@}" >/dev/null 2>/dev/null
}

install_dependencies() {
  if ! find_in_conda_env nir-coloring; then
    conda env create --file "$ROOT_DIR/environment.yml";
    conda develop -n nir-coloring nir-coloring
  fi;

  if ! find_in_conda_env cycleGAN; then
    conda env create --file "$ROOT_DIR/cycle-gan/environment.yml";
  fi;
}

install_dependencies

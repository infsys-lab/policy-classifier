#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare_submodules.sh [-h|--help]

This script clones the 'policy-classifier-data' repository

optional arguments:
  -h, --help  show this help message and exit
EOF
}

# check for help
check_help() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

# define main function
main() {
  git submodule update --init --recursive
}

# execute all functions
check_help "$@"
main

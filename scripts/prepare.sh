#!/usr/bin/env bash
set -e

# usage function
usage() {
  cat <<EOF
usage: prepare.sh [-h|--help]

This script downloads and unzips necessary data to
train the privacy policy classifier

optional arguments:
  -h, --help  show this help message and exit
EOF
}

# check for help
parser() {
  for arg; do
    if [ "$arg" == "--help" ] || [ "$arg" == "-h" ]; then
      usage
      exit 0
    fi
  done
}

# download and prepare privacy policies
privacy_policies() {
  local directory="./data"
  wget -N -P "$directory" "https://privacypolicies.cs.princeton.edu/data-release/data/classifier_data.tar.gz"
  tar -zxvf "$directory/classifier_data.tar.gz" -C "$directory" --strip-components 1 "dataset/1301_dataset.csv"
}

# define main function
main() {
  privacy_policies
}

# execute all functions
parser "$@"
main

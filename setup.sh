#!/usr/bin/bash
# A simple script to easily setup directories and download the dataset.

while getopts ":hdom" opt; do
  case $opt in
    h)
      echo "
        -d Download the dataset.
        -o Setup the out subdirectories.
        -m Download the model for the demo." >&2
      ;;
    d)
      kaggle competitions download -c birdclef-2021
      unzip birdclef-2021 -d data
      ;;
    o)
      wget https://edodema.xyz/files/out.tar.xz
      tar xf out.tar.xz
      rm out.tar.xz
      ;;
    m)
      cd models || exit
      wget https://edodema.xyz/files/CNNRes2GRU1FC1.ckpt.tar.xz
      tar xf CNNRes2GRU1FC1.ckpt.tar.xz
      rm CNNRes2GRU1FC1.ckpt.tar.xz
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done
#!/usr/bin/bash
# A simple script to easily setup directories and download the dataset.

while getopts ":hdom" opt; do
  case $opt in
    h)
      echo "./setup.sh [-dom]
        -d Download the dataset.
        -o Setup the out subdirectories.
        -m Download the model for the demo." >&2
      ;;
    d)
      kaggle competitions download -c birdclef-2021
      unzip birdclef-2021.zip -d data
      rm birdclef-2021.zip
      ;;
    o)
      wget https://edodema.xyz/files/out.tar.xz
      tar xf out.tar.xz
      rm out.tar.xz
      ;;
    m)
      wget https://edodema.xyz/files/CNNRes2GRU1FC1.ckpt.tar.xz
      tar xf CNNRes2GRU1FC1.ckpt.tar.xz -C models/
      rm CNNRes2GRU1FC1.ckpt.tar.xz
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done

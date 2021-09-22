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
      mkdir out || return 0
      cd out || exit
      mkdir precomputed split_datasets vocabs || return 0

      # Precomputed.
      cd precomputed || exit
      mkdir train val test || return 0

      ## Train
      cd train || exit
      mkdir birdcalls soundscapes joint || return 0
      cd .. || exit

      ## Val
      cd val || exit
      mkdir birdcalls soundscapes joint || return 0
      cd .. || exit

      ## Test
      cd test || exit
      mkdir birdcalls soundscapes joint || return 0
      cd .. || exit
      cd .. || exit

      # Split datasets
      mkdir split_datasets || return 0
      cd split_datasets || exit
      mkdir train val test || return 0
      cd .. || exit
      ;;
    m)
      cd models || exit
      wget http://ginonardella.xyz/files/CNNRes2GRU1FC1.ckpt.tar.xz
      tar xf CNNRes2GRU1FC1.ckpt.tar.xz
      rm CNNRes2GRU1FC1.ckpt.tar.xz
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done
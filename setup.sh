#!/usr/bin/bash
# A simple script to easily setup directories and download the dataset.

while getopts ":hdo" opt; do
  case $opt in
    h)
      echo "
        -d Download the dataset.
        -o Setup the out subdirectories." >&2
      ;;
    d)
      kaggle competitions download -c birdclef-2021
      unzip birdclef-2021 -d data
      ;;
    o)
      mkdir out
      cd out || exit
      mkdir precomputed split_datasets vocabs

      # Precomputed.
      cd precomputed || exit
      mkdir train val test

      ## Train
      cd train || exit
      mkdir birdcalls soundscapes joint
      cd .. || exit

      ## Val
      cd val || exit
      mkdir birdcalls soundscapes joint
      cd .. || exit

      ## Test
      cd test || exit
      mkdir birdcalls soundscapes joint
      cd .. || exit
      cd .. || exit

      # Split datasets
      mkdir split_datasets
      cd split_datasets || exit
      mkdir train val test
      cd .. || exit
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      ;;
  esac
done
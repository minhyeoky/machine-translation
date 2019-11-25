#!/usr/bin/env bash

# Default arguments
USE_DOCKER=false
USE_GPU=""
TAG="nmt"
DOCKER_DIR="/tf"
MODEL_NAME="seq2seq"

while getopts "g:dj:v:m:" opt; do
  case "$opt" in
  d)
    USE_DOCKER=true
    ;;
  g)
    USE_GPU="$OPTARG"
    ;;
  j)
    CONFIG_JSON="$OPTARG"
    ;;
  v)
    VOLUME_DIR="$OPTARG"
    ;;
  m)
    MODEL_NAME="$OPTARG"
    ;;
  *)
    echo "Invalid arguments are provieded"
    ;;
  esac
done

if $USE_DOCKER; then
  echo "Running in docker container"

  # integrity check
  if [[ -z "$CONFIG_JSON" ]]; then
    echo "  CONFIG_JSON is not set"
  else
    echo "  Configuration: $CONFIG_JSON"
  fi
  if [[ -z "$VOLUME_DIR" ]]; then
    echo "  VOLUME_DIR is not set"
  else
    echo "  Mount volume: $VOLUME_DIR"
  fi

  # Run
  docker build . --tag=$TAG
  if [[ -n $USE_GPU ]]; then
    echo "    Running with GPU $USE_GPU"
    docker run \
      --runtime=nvidia \
      -e NVIDIA_VISIBLE_DEVICES=$USE_GPU \
      -e DOCKER_DIR=$DOCKER_DIR \
      -v "$VOLUME_DIR":$DOCKER_DIR \
      $TAG \
      python train.py \
      --config_json=$CONFIG_JSON \
      --data_path=data/input/aihub_kor-eng/1.구어체.xlsx \
      --model="$MODEL_NAME"
  else
    echo "    Running without GPU"
    docker run \
      -e DOCKER_DIR=$DOCKER_DIR \
      -v "$VOLUME_DIR":$DOCKER_DIR \
      $TAG \
      python train.py \
      --config_json=$CONFIG_JSON \
      --data_path=data/input/aihub_kor-eng/1.구어체.xlsx \
      --model="$MODEL_NAME"
  fi
else
  # Run locally
  python3 train.py \
    --config_json=$CONFIG_JSON \
    --data_path=data/input/aihub_kor-eng/1.구어체.xlsx \
    --model="$MODEL_NAME"
fi

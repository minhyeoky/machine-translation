#!/usr/bin/env bash

# Default arguments
USE_DOCKER=false
USE_GPU=""
TAG="nmt"
DOCKER_DIR="/tf"
MODEL_NAME="seq2seq"
DATA_PATH=data/input/aihub_kor-eng/1.구어체.xlsx

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



if [[ $MODEL_NAME == "seq2seq" ]]; then
  RUN_TRAIN="python train_seq2seq.py --config_json=$DOCKER_DIR/$CONFIG_JSON --data_path=$DATA_PATH"
elif [[ $MODEL_NAME == "bahdanau" ]]; then
  RUN_TRAIN="python train_bahdanau.py --config_json=$DOCKER_DIR/$CONFIG_JSON --data_path=$DATA_PATH"
elif [[ $MODEL_NAME == "seq2seq_bidirectional" ]]; then
  RUN_TRAIN="python train_seq2seq_bidirectional.py --config_json=$DOCKER_DIR/$CONFIG_JSON --data_path=$DATA_PATH"
elif [[ $MODEL_NAME == "bahdanau_bidirectional" ]]; then
  RUN_TRAIN="python train_bahdanau_bidirectional.py --config_json=$DOCKER_DIR/$CONFIG_JSON --data_path=$DATA_PATH"
elif [[ $MODEL_NAME == "transformer" ]]; then
  RUN_TRAIN="python train_transformer.py --config_json=$DOCKER_DIR/$CONFIG_JSON --data_path=$DATA_PATH"
fi


if $USE_DOCKER; then
  echo "Running in docker container"

  echo "  MODEL: $MODEL_NAME"
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
      -e NVIDIA_VISIBLE_DEVICES="$USE_GPU" \
      -e DOCKER_DIR=$DOCKER_DIR \
      -v "$VOLUME_DIR":$DOCKER_DIR \
      $TAG \
      $RUN_TRAIN

  else
    echo "    Running without GPU"
    docker run \
      -e DOCKER_DIR=$DOCKER_DIR \
      -v "$VOLUME_DIR":$DOCKER_DIR \
      $TAG \
      $RUN_TRAIN
  fi
else
  # Run locally
  $RUN_TRAIN
fi

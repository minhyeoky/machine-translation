#!/usr/bin/env bash

config=$1
nmt_dir=$2
num_gpu=$3
docker_dir=/tf
tag=nmt
py=train.py

docker build . --tag="${tag}"
docker run --runtime=nvidia -e docker_dir=${docker_dir} -e NVIDIA_VISIBLE_DEVICES="${num_gpu}" -v "${nmt_dir}":${docker_dir} ${tag} \
  python ${docker_dir}/${py} --config_json=${docker_dir}/"${config}" --data_path=${docker_dir}/data/input/aihub_kor-eng/1.구어체.xlsx

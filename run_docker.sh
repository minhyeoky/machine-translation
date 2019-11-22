#!/usr/bin/env bash

config=$1
nmt_dir=$2
docker_dir=/tf
tag=nmt
py=train.py

docker build . --tag="${tag}"
docker run -e docker_dir=${docker_dir} -v "${nmt_dir}":${docker_dir} ${tag} \
  python ${docker_dir}/${py} --config_json=${docker_dir}/"${config}" --data_path=${docker_dir}/data/input/aihub_kor-eng/1.구어체.xlsx

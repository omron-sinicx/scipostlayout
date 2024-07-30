#/bin/bash

docker run --shm-size=1g -it --rm --gpus all --name scipostlayout \
 --volume $(pwd):/scipostlayout shoheita/scipostlayout:latest

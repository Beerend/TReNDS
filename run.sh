#!/bin/bash

docker build --no-cache -t capita_selecta_cvbm -f config/capita_selecta_cvbm.Dockerfile .
docker run -ti -v ${PWD}:/usr/local/bin/capita_selecta_cvbm -p 8888:8888 capita_selecta_cvbm
#!/bin/bash

docker start superpython
docker cp "$1" superpython:/home/zdjs/myfile
docker exec superpython symbolic "/home/zdjs/myfile"
docker exec superpython rm "/home/zdjs/myfile"

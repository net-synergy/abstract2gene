#!/usr/bin/env bash

cd $(dirname $0)

docker run --rm -v $(pwd):/home/docker/package -v $XDG_DATA_HOME:/home/docker/.local/share -v $XDG_CACHE_HOME:/home/docker/.cache -w /home/docker/package -it abstract2gene:nix ipython

#!/usr/bin/env bash

cd $(dirname $0)

docker run --rm -v $(pwd):$HOME -w $HOME -it abstract2gene:nix ipython

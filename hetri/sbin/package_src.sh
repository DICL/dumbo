#!/usr/bin/env bash
cd $(dirname $(readlink -f $0))
cd ..

tools/sbt assembly

tar zcvf hetri-0.1-src.tar.gz src/main/scala bin tools do_hetri.sh README.md simple.edge Makefile

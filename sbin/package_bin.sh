#!/usr/bin/env bash
cd $(dirname $(readlink -f $0))
cd ..

tools/sbt assembly

tar zcvf hetri-0.1.tar.gz bin tools do_hetri.sh README.md simple.edge Makefile

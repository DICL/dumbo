#!/usr/bin/env bash
cd $(dirname $(readlink -f $0))

hadoop fs -rm -r simple.edge
hadoop fs -put simple.edge simple.edge

hadoop jar bin/hetri-0.1.jar hetri.HeTri \
                                       -Dallocator=mlc \
                                       -DnumColors=3 \
                                       simple.edge

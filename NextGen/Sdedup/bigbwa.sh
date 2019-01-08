#!/bin/bash
bigbwaConf="$(cat bigbwa_config)"
#echo "yarn jar ./BigBWA/target/BigBWA-2.1.jar com.github.bigbwa.BigBWA -D mapreduce.input.fileinputformat.split.minsize=123641127 "
#echo "-D mapreduce.input.fileinputformat.split.maxsize=123641127"
#echo "-D mapreduce.map.memory.mb=7500"
#echo "-w \"-F $1 "$bigbwaConf"\" "
#echo "-m -p --index $2 $3 $4"

yarn jar ./BigBWA/target/BigBWA-2.1.jar com.github.bigbwa.BigBWA -D mapreduce.input.fileinputformat.split.minsize=123641127 \
-D mapreduce.input.fileinputformat.split.maxsize=123641127 \
-D mapreduce.map.memory.mb=7500 \
-w \""$bigbwaConf"\" \
-m -p --index $1 $2 $3

./sendEnd

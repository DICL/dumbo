#!/bin/bash

# build hadoop
mvn package -e -Pdist -DskipTests -Dtar

# build GPU-Monitor for Hybrid Hadoop
cd ./csrc/GPUMonitor_JNI
make
cd -

# copy conf files to dist directory
cd hadoop-dist/target/hadoop-2.7.3
ln -s etc/hadoop conf
cp ../../../conf/* conf

# copy GPU-Monitor native library
mkdir -p lib/native
cd lib/native
ln -s ../../../../../csrc/GPUMonitor_JNI/libJNI_GPUMonitor.so   


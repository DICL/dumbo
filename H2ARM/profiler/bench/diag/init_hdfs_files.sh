#!/bin/bash

HADOOP_PREFIX="/home/jojaeeon/bigdata/hadoop/hadoop-dist/target/hadoop-2.6.0-cdh5.11.1"
HDFS_DIR="/file_access_app"
#FILE_SIZE="134217728"
FILE_SIZE=67108864
NUM_FILES=512
NODES=(\
TRUSTY100
TRUSTY101
TRUSTY102
TRUSTY104
TRUSTY105
TRUSTY106
)

if ! [[ -d ${HADOOP_PREFIX} ]]; then
    echo "HADOOP_PREFIX=${HADOOP_PREFIX} does not exists!"
    exit 10
fi

$HADOOP_PREFIX/bin/hdfs dfs -rm -r -f ${HDFS_DIR}
$HADOOP_PREFIX/bin/hdfs dfs -mkdir ${HDFS_DIR}

PIDS=()
NID=0
for i in $( seq 0 $((NUM_FILES - 1)) ); do
    MAX_FID=$((NUM_FILES - 1))
    FID=$(printf "%0${#MAX_FID}d" $i)
    NODE=${NODES[${NID}]}
    FILE_PATH=${HDFS_DIR}/${FID}
    echo ${NODE} ${FILE_PATH} &
    ssh ${NODE} "head -c ${FILE_SIZE} </dev/urandom | ${HADOOP_PREFIX}/bin/hdfs dfs -put - ${FILE_PATH}" &
    PIDS+=($!)
    NID=$(((NID + 1) % ${#NODES[@]}))
    if [[ ${NID} -eq 0 ]]; then
        for PID in ${PIDS[@]}; do
            wait ${PID}
            echo wait ${PID}
        done
        PIDS=()
    fi
done

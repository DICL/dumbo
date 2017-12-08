#!/usr/bin/env bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`

. "$bin"/hadoop-config.sh

"$HADOOP_HOME/bin/hadoop" jobtracker >"${HADOOP_LOG_DIR}/hadoop-${USER}-jobtracker-${HOSTNAME}.log" 2>&1 &
echo -n $! >"/tmp/hadoop-${USER}-jobtracker.pid"

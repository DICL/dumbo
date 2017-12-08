#!/usr/bin/env bash

bin=`dirname "$0"`
bin=`cd "$bin"; pwd`

. "$bin"/hadoop-config.sh

kill -9 $(cat /tmp/hadoop-${USER}-jobtracker.pid)

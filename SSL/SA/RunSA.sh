#!/bin/bash

chmod 755 $0
chmod 755 SA_Exec

export CLASSPATH=`pwd`/weka.jar:$CLASSPATH

TEMP=1
THRES=0.2
CF=0.8
SETUP=Setup.dat
APP=App.dat

g++ SimulatedAnnealing.cpp -o SA_Exec

echo $TEMP $THRES $CF $SETUP $APP
rm MLpredict/Temp/* MLpredict/Temp_FM/* 2> /dev/null
rm *.sample 2> /dev/null
./SA_Exec $TEMP $THRES $CF $SETUP $APP > log
pids=(`ps ax | grep sh | grep SAhelper | awk '{print $1}'`)
for pid in ${pids[@]} ; do
	kill -9 $pid
done

rm MLpredict/Temp/* MLpredict/Temp_FM/*
rm *.sample

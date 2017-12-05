#!/bin/bash
#BigBWA

#$1-$2 => SparkServer, HadoopServer

ssh $2 "nohup Sdedup-BigBWA.sh"
ssh $1 "nohup Sdedup-Spark.sh"
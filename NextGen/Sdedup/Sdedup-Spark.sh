#!/bin/bash
#Spark-submit

#Spark Workers
#server[0]="192.168.0.100"
#server[1]="192.168.0.1"
#...

for (( i=0 ; i<${#server[*]} ; i++ )); do
    ssh ${server[$i]} "nohup java -cp . EchoServer | samblaster --ignoreUnmated > sam.out 2> sam.log"	
done

#$1~$4 => 카프카 서버, 토픽, 파티셔닝방법, 워커
spark-submit [parameters] path/to/jar [Kafka_Servers] [Kafka_topic] [Spark_workers] [Partitioning method (1-qname, 2-rname, 3-pos, 4-rname+pos)]
#(e.g) $spark-submit --driver-cores 16 --driver-memory 30g --executor-memory 30g --class lab.test001 --master spark://114.70.216.166:7077 ./target/spark0331-1.0-SNAPSHOT-jar-with-dependencies.jar 172.31.7.168:9092,172.31.10.122:9092,172.31.14.59:9092 mytopic 172.31.5.87,172.31.11.122,172.31.12.146 1
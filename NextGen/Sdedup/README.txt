---------------------------------------------------------

실행환경

1. kafka가 설치된 서버
2. Hadoop Yarn과 BigBWA, FullSAM이 설치된 서버
3. Spark Standalone 서버

! Spark Worker노드는 samblaster와 EchoServer가 설치되어있어야 함

---------------------------------------------------------


실행방법

1. BigBWA를 실행할 Hadoop서버에 Sdedup-BigBWA.sh 을 생성합니다.
=> Sdedup-BigBWA.sh 에 BigBWA 실행 파라미터를 입력합니다.

2. Spark-submit을 실행 할 Spark 서버에 Sdedup-Spark.sh 파일을 생성합니다.
=> Sdedup-Spark.sh 에 spark-submit 실행 파라미터를 입력합니다.

쉘 파일 Sdedup.sh 을 실행해주세요.


---------------------------------------------------------


! 필요한 부분을 수정해서 사용하세요.
! BigBWA는 파라미터가 많아 https://github.com/citiususc/BigBWA 에서 파라미터를 확인해주세요.
! Sdedup 실행시 worker수와 executor 수를 같게 해줘야 합니다.
! 파라미터는 Sdedup-BigBWA.sh, Sdedup-Spark.sh 파일을 수정해서 입력합니다.



Bigbwa - 
(e.g) $yarn jar BigBWA-2.1.jar com.github.bigbwa.BigBWA -D mapreduce.input.fileinputformat.split.minsize=123641127 -D mapreduce.input.fileinputformat.split.maxsize=123641127 -D mapreduce.map.memory.mb=7500 -w "-R @RG\tID:foo\tLB:bar\tPL:illumina\tPU:illumina\tSM:ERR000589 -t 2"-m -p --index /mnt/data/hg38.fa -r ERR000585.fqBD ExitERR000585

fullSam - [Directory/Sam/Files/] [kafka_topic] [kafka_servers]

Sdedup - [Kafka_Servers] [Kafka_topic] [Spark_workers] [mapping method (1-qname, 2-rname, 3-pos, 4-rname+pos)]
(e.g) $spark-submit --driver-cores 16 --driver-memory 30g --executor-memory 30g --class lab.test001 --master spark://114.70.216.166:7077 ./target/spark0331-1.0-SNAPSHOT-jar-with-dependencies.jar 172.31.7.168:9092,172.31.10.122:9092,172.31.14.59:9092 head,read 172.31.5.87,172.31.11.122,172.31.12.146 1


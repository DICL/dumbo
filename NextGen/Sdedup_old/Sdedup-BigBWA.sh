#!/bin/bash
#BigBWA

#BigBWA 실행
#(e.g) yarn jar BigBWA-2.1.jar com.github.bigbwa.BigBWA -D mapreduce.input.fileinputformat.split.minsize=123641127 -D mapreduce.input.fileinputformat.split.maxsize=123641127 -D mapreduce.map.memory.mb=7500 -w "-R @RG\tID:foo\tLB:bar\tPL:illumina\tPU:illumina\tSM:ERR000589 -t 2"-m -p --index /mnt/data/hg38.fa -r ERR000585.fqBD ExitERR000585

python fullsam.py [Path to Sams] [Kafka topic]
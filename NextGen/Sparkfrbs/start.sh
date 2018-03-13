#!/bin/bash
#fa=/home/dblab/Desktop/reference/hg38.fa
#bs=/home/dblab/BioInfoData/bamfile/bam-file-list
#outp=/home/dblab/works/sparkfrbs/out
#ptn=4
#region=10000000
#start_time="$(date -u +%s)"
#spark-submit --master spark://ubuntu01:7077 --class dblab.test0 target/sparkfrbs-1.0-SNAPSHOT.jar /usr/local/bin/freebayes $fa $bs $outp 4 10000000
#end_time="$(date -u +%s)"
#elapsed="$(($end_time-$start_time))"
#echo "Total of $elapsed seconds elapsed for process"
#/home/dblab/sparkfrbs/outputfile/ 10 100
#file:///home/dblab/sparkfrbs/datafile/kitty.sam \
master=$1
ref=$2
bamlist=$3
outdir=$4
partnum=$5
region=$6
#spark-submit --master $master --class dblab.sparkfrbs target/sparkfrbs-1.0-SNAPSHOT.jar $ref $bamlist $outdir $partnum $region
echo "spark-submit --master $master --class dblab.sparkfrbs target/sparkfrbs-1.0-SNAPSHOT.jar $ref $bamlist $outdir $partnum $region"

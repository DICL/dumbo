#!/bin/sh

echo $#

if [ $# -lt 5 ]
then
        echo "USAGE : run1.sh [WORK DIR(OUTPUT PATH)] [REFERNCE FILE] [FASTQ PATH (HDFS, PAIRED ONLY)] [HDFS WORK DIR] [VCF OUT DIR] (OPT. ConfigFile) (OPT. Parallel optition)"
        exit 1
fi

if [ $# -eq 5 ]
then
        echo "run as single mode"
        source "./resource_config"
fi

if [ $# -eq 6 ]
then
        echo "load $6 config file"        
        source "./$6"
fi

broker() 
{
  echo "$ZIN $ZOUT &"
  ./sdedup_broker $ZIN $ZOUT &
  broker_pid=$!
}

sdedup() 
{
  spark-submit --class V3.Sdedup --master yarn --deploy-mode client --driver-memory $sDriverMem --executor-cores $sExecutorCores --executor-memory $sExecutorMem --num-executors $sNumExecutors \
  --conf "spark.driver.extraJavaOptions=-XX:+UseG1GC" \
  --conf "spark.executor.extraJavaOptions=-XX:+UseG1GC" \
  ./Sdedup/target/Sdedup-1.0-SNAPSHOT-jar-with-dependencies.jar $ZOUT $sDstreamSize $sPartitionNumb $1
}

merge() 
{
spark-submit --class V3.SMerger2 --master yarn --deploy-mode client --driver-memory $mDriverMem --executor-cores $mExecutorCores --executor-memory $mExecutorMem --num-executors $mNumExecutors  ./Sdedup/target/Sdedup-1.0-SNAPSHOT-jar-with-dependencies.jar $1 $mSamtoolsPath bam
}

freebayes() 
{
spark-submit --class V3.sparkfrbs_v2 --master yarn --deploy-mode client --driver-memory $fDriverMem --executor-cores $fExecutorCores --executor-memory $fExecutorMem --num-executors $fNumExecutors ./Sdedup/target/Sdedup-1.0-SNAPSHOT-jar-with-dependencies.jar $fFreebayesPath $fSamtoolsPath $1 $2 $3 $fPartNum $fRegionSize
#n("usage : FreebayesPath SamtoolsPath ReferencefilePath WorkDir VcfOutputDirPath PartitionNumb RegionSize")
}


OUT=$1
RG=$2
FQ=$3
HDFSBB=$4
VCFOUT=$5


#broker
#./bigbwa.sh $RG $FQ $HDFSBB > bigbwa.log 2>&1 &
#sdedup $OUT
#merge $OUT

if [ $# -eq 7 ]
then
        echo "run as parallel"
        echo "touch file $7"
        touch $7.done
        exit 1
fi

#freebayes $RG $OUT $VCFOUT

ls $VCFOUT/*.vcf > $VCFOUT/list.txt
$fBcftoolsPath concat -f $VCFOUT/list.txt -o $VCFOUT/merged.vcf
kill -9 $broker_pid

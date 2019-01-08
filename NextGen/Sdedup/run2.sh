#/bin/sh
echo $#
if [ $# -lt 9 ]
then
        echo "USAGE : run2.sh [WORK DIR 1(OUTPUT PATH)] [WORK DIR 2(OUTPUT PATH)] [REFERNCE FILE] [FASTQ PATH 1(HDFS, PAIRED ONLY)] [HDFS WORK DIR 1] [FASTQ PATH 2(HDFS, PAIRED ONLY)] [HDFS WORK DIR 2] [CONFIG FILE] [VCF OUT DIR]"
        exit 1
fi

source $8

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
spark-submit --class V3.sparkfrbs_v3 --master yarn --deploy-mode client --driver-memory $fDriverMem --executor-cores $fExecutorCores --executor-memory $fExecutorMem --num-executors $fNumExecutors ./Sdedup/target/Sdedup-1.0-SNAPSHOT-jar-with-dependencies.jar $fFreebayesPath $fSamtoolsPath $1 $2 $3 $4 $fPartNum $fRegionSize
#n("usage : FreebayesPath SamtoolsPath ReferencefilePath WorkDir1 WorkDir2 VcfOutputDirPath PartitionNumb RegionSize")
}

OUT1=$1
OUT2=$2
RG=$3
FQ=$4
HDFSBB=$5
FQ2=$6
HDFSBB2=$7
VCFOUT=$8

echo "1/2 run"

#broker
#./bigbwa.sh $RG $FQ &
#sdedup $OUT1
#merge $OUT1

echo "2/2 run"

#broker
#./bigbwa.sh $RG $FQ2 &
#sdedup $OUT2
#merge $OUT2

#/lustre/hadoop/user/dblab/testout/spark-out2/streamdir.done
freebayes $RG $OUT1 $OUT2 $VCFOUT

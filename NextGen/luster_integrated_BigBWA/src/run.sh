#/bin/sh
if [ $# -lt 5 ]
then
        echo "USAGE : run.sh <# of partitions> <# of threads per mapper> <input_1> <input_2> <outputdir> [<reference genome>]"
        echo "   input output file location is relative path from HDFS user home directory"
        exit 1
fi


LUSTRE_ADAPTER=../lustrefs/target/lustrefs-hadoop-0.9.1.jar

RG=/lustre/scratch/bwauser/Data/HumanBase/hg19.fa
if [ $# -gt 5 ]
then
	if [ -f $6 ]
	then 
		RG=$6
	else
		echo "The reference genome file is not found. Use the default one"
	fi
fi
echo "Reference Genome = "${RG}

MOUNT_LOC=`hdfs getconf -confKey fs.lustrefs.mount`
MOUNT_LOC=${MOUNT_LOC}/user/${USER}/$5
PREFIX=${MOUNT_LOC}/out/Output
OUTPUT=${MOUNT_LOC}/merged.sam
SHARED_TMP=`hdfs getconf -confKey fs.lustrefs.shared_tmp.dir`

hdfs dfs -rm -r -f ${SHARED_TMP}/*

time hadoop jar BigBWA.jar -archives bwa.zip -libjars ${LUSTRE_ADAPTER},BigBWA.jar -partitions $1 -threads $2 -algorithm mem -reads paired -index ${RG} $3 $5/out
hdfs dfs -rm -r -f $5/out

time hadoop jar BigBWA.jar -archives bwa.zip -libjars ${LUSTRE_ADAPTER},BigBWA.jar -partitions $1 -threads $2 -algorithm mem -reads paired -index ${RG} $4 $5/out

NP=`ls -l ${PREFIX}* | wc -l`

time mpirun -np $NP -hostfile ${HADOOP_HOME}/etc/hadoop/slaves --map-by node  ./reduce ${PREFIX} ${OUTPUT}

hdfs dfs -rm -r -f $5/out

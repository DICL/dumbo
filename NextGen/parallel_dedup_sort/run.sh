#/bin/sh
if [ $# -lt 3 ]
then
        echo "USAGE : run.sh <# of parallel processes> <bigbwa_result_file_prefix> <outputdir> <hostlistfile>"
	echo "	 <example>"
	echo "    1) ./run.sh /lustre/hadoop/user/bwauser/Tumor2/sams/Output /lustre/hadoop/user/bwauser/output2 hf"
	echo "    2) ./run.sh /lustre/hadoop/user/bwauser/ERR000589/sams/Output /lustre/hadoop/user/bwauser/output2 hf"
        exit 1
fi

BIN=${PWD}
SAMBLASTER=/home/bwauser/samblaster/samblaster
SAMTOOLS=/home/bwauser/samtools-1.3/bin/samtools 

NBR=`ls -l $1*.sam | wc -l`
TW=`cat $3 | wc -l`

TMP_DIR=$2/tmpfiles

echo Preparing temp directory : ${TMP_DIR}
if [ ! -e "${TMP_DIR}" ]
then
	mkdir ${TMP_DIR}
fi

if [ ! -d "${TMP_DIR}" ] 
then
	echo "FAIL to create temp directory. The non-directory file ${TMP_DIR} already exists."
	exit 1;
fi

echo "Start ${TW} sinks for parallel instance of samblaster"
parallel -j ${TW} -a $3 --linebuffer ssh {} \""${BIN}/sink ${NBR} {#} | ${SAMBLASTER} | ${BIN}/d2s {#} ${TW} ${TMP_DIR}"\" &

echo "Start shuffle data from ${NBR} KBigBWA output files to samblaster sinks"
time mpirun -np ${NBR} -hostfile ${HADOOP_HOME}/etc/hadoop/slaves --map-by node ${BIN}/s4dmpi $1 $3
wait

echo "Running ${TW} samtools from tmp files"
time parallel -j ${TW} -a $3 ssh {} \""cat ${TMP_DIR}/fsplit_tmp.header.{#} ${TMP_DIR}/fsplit_tmp_data.{#}.* | ${SAMTOOLS} sort -@ 4 -T {#} -o $2/DS{#}.bam"\"

echo "Merging bam file to $2/merged.bam"
time mpirun -np ${NBR} -hostfile ${HADOOP_HOME}/etc/hadoop/slaves --map-by node ${BIN}/merge_bam $2/DS $2/merged.bam 1

echo "Making index"
time ${SAMTOOLS} index $2/merged.bam

echo "Removing temp files"
rm -f $2/DS*.bam
rm -f ${TMP_DIR}/*
rmdir ${TMP_DIR}

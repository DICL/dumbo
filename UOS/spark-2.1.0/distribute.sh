SPARK_TGZ="$SPARK_HOME/spark-1-bin-spark-180122-pm.tgz"
SPARK_CONF="/home/qyu/Kit/Spark"

#psh node02-node16 -s "scp -r node01:$SPARK_TGZ $SPARK_CONF"
#echo "TGZ copy done"
#psh node02-node16 -s "tar -xzf ${SPARK_CONF}/spark-1-bin-spark-180122-pm.tgz -C ${SPARK_CONF}"
#echo "tar tgz done"
psh node02-node16 -s "rm ${SPARK_CONF}/spark-2.1.0 -r"
echo "rm lasted version"
psh node02-node16 -s "mv ${SPARK_CONF}/spark-1-bin-spark-180122-pm ${SPARK_CONF}/spark-2.1.0"
echo "mv latest version"
psh node02-node16 -s "rm ${SPARK_CONF}/spark-1-bin-spark-180122-pm.tgz"
echo "rm tar"

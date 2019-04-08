SPARK_CONF="$SPARK_HOME/conf"
for i in `seq 2 16`;
do
		if [ $i -lt 10 ]
		then
				scp * node0$i:$SPARK_CONF
				psh node0$i "sed -i 's/SPARK_LOCAL_IP=10.0.0.1/SPARK_LOCAL_IP=10.0.0.$i/g' $SPARK_CONF/spark-env.sh"
		else
				scp * node$i:$SPARK_CONF
				psh node$i "sed -i 's/SPARK_LOCAL_IP=10.0.0.1/SPARK_LOCAL_IP=10.0.0.$i/g' $SPARK_CONF/spark-env.sh"
		fi			
done

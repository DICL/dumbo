package org.kisti.moha;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Properties;

public class MOHA_KafkaStop {
	private static MOHA_Zookeeper zk;

	public static void main(String[] args) {
		MOHA_KafkaStop stop = new MOHA_KafkaStop();
		stop.run();
	}

	public MOHA_KafkaStop() {
		Properties prop = new Properties();
		/* Loading MOHA.Conf File */
		String kafka_libs = "/usr/hdp/kafka_2.11-0.9.0.0/libs/*";
		String kafkaVersion = "kafka_2.11-0.10.1.0";
		String kafkaClusterId = "KafkaCluster";
		try {
			prop.load(new FileInputStream("conf/MOHA.conf"));
			kafka_libs = prop.getProperty("MOHA.dependencies.kafka.libs");
			kafkaVersion = prop.getProperty("MOHA.dependencies.kafka.version");
			kafkaClusterId = prop.getProperty("MOHA.kafka.cluster.id");
			System.out.println(kafka_libs);
			System.out.println(kafkaVersion);
			System.out.println(kafkaClusterId);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		zk = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_KAFKA , kafkaClusterId);
	}

	public void run() {

		zk.setRequests(true);

	}
}

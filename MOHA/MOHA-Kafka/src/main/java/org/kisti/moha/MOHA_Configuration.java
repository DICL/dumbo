package org.kisti.moha;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Properties;

public class MOHA_Configuration {

	private String kafkaLibsDirs;
	private String kafkaVersion;
	private String kafkaClusterId;
	private String debugQueueName;
	private String mysqlLogEnable;
	private String kafkaDebugEnable;
	private String zookeeperConnect;
	private String bootstrapServers;

	public String getZookeeperConnect() {
		return zookeeperConnect;
	}

	public void setZookeeperConnect(String zookeeperConnect) {
		this.zookeeperConnect = zookeeperConnect;
	}

	public String getBootstrapServers() {
		return bootstrapServers;
	}

	public void setBootstrapServers(String bootstrapServers) {
		this.bootstrapServers = bootstrapServers;
	}

	public MOHA_Configuration(String confDir) {
		Properties prop = new Properties();
		init();
		try {
			prop.load(new FileInputStream(confDir));

			setDebugQueueName(prop.getProperty("MOHA.kafka.debug.queue.name", "debug"));
			setKafkaClusterId(prop.getProperty("MOHA.kafka.cluster.id", "Kafka/Cluster1"));
			setKafkaVersion(prop.getProperty("MOHA.dependencies.kafka.version", "kafka_2.11-0.10.1.0"));
			setKafkaLibsDirs(prop.getProperty("MOHA.dependencies.kafka.libs", "/usr/hdp/kafka_2.11-0.10.1.0/libs/*"));
			setEnableKafkaDebug(prop.getProperty("MOHA.kafka.debug.enable", "false"));
			setEnableMysqlLog(prop.getProperty("MOHA.mysql.log.enable", "false"));
			setBootstrapServers(prop.getProperty("MOHA.zookeeper.bootstrap.servers", "localhost:9092"));
			setZookeeperConnect(prop.getProperty("MOHA.zookeeper.connect", "localhost:2181"));

		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public MOHA_Configuration() {
		init();
	}

	public MOHA_Configuration(Map<String, String> getenv) {

		init();

		if (getenv.containsKey(MOHA_Properties.KAKFA_VERSION)) {
			setKafkaVersion(getenv.get(MOHA_Properties.KAKFA_VERSION));
		}
		if (getenv.containsKey(MOHA_Properties.KAFKA_CLUSTER_ID)) {
			setKafkaClusterId(getenv.get(MOHA_Properties.KAFKA_CLUSTER_ID));
		}

		if (getenv.containsKey(MOHA_Properties.KAFKA_DEBUG_ENABLE)) {
			setEnableKafkaDebug(getenv.get(MOHA_Properties.KAFKA_DEBUG_ENABLE));
		}

		if (getenv.containsKey(MOHA_Properties.KAFKA_DEBUG_QUEUE_NAME)) {
			setDebugQueueName(getenv.get(MOHA_Properties.KAFKA_DEBUG_QUEUE_NAME));
		}

		if (getenv.containsKey(MOHA_Properties.MYSQL_DEBUG_ENABLE)) {
			setEnableMysqlLog(getenv.get(MOHA_Properties.MYSQL_DEBUG_ENABLE));
		}

		if (getenv.containsKey(MOHA_Properties.CONF_ZOOKEEPER_CONNECT)) {
			setZookeeperConnect(getenv.get(MOHA_Properties.CONF_ZOOKEEPER_CONNECT));
		}
		if (getenv.containsKey(MOHA_Properties.CONF_ZOOKEEPER_BOOTSTRAP_SERVER)) {
			setBootstrapServers(getenv.get(MOHA_Properties.CONF_ZOOKEEPER_BOOTSTRAP_SERVER));
		}

	}

	private void init() {
		// initiate variables
		setDebugQueueName("debug");
		setKafkaClusterId("Kafka/Cluster1");
		setKafkaVersion("kafka_2.11-0.10.1.0");
		setKafkaLibsDirs("/usr/hdp/kafka_2.11-0.10.1.0/libs/*");
		setEnableKafkaDebug("false");
		setEnableMysqlLog("false");
		setZookeeperConnect("localhost:2181");
		setBootstrapServers("localhost:9092");
	}

	@Override
	public String toString() {
		return "MOHA_Configuration [kafkaLibsDirs=" + kafkaLibsDirs + ", kafkaVersion=" + kafkaVersion
				+ ", kafkaClusterId=" + kafkaClusterId + ", debugQueueName=" + debugQueueName + ", mysqlLogEnable="
				+ mysqlLogEnable + ", kafkaDebugEnable=" + kafkaDebugEnable + ", zookeeperConnect=" + zookeeperConnect
				+ ", bootstrapServers=" + bootstrapServers + "]";
	}

	// get configuration information
	List<String> getInfo() {
		List<String> conf = new ArrayList<>();
		conf.add(getKafkaLibsDirs());
		conf.add(getKafkaVersion());
		conf.add(getKafkaClusterId());
		conf.add(getDebugQueueName());
		conf.add(getKafkaDebugEnable());
		conf.add(getMysqlLogEnable());
		return conf;
	}

	public String getMysqlLogEnable() {
		return mysqlLogEnable;
	}

	public void setEnableMysqlLog(String enableMysqlLog) {
		this.mysqlLogEnable = enableMysqlLog;
	}

	public String getKafkaDebugEnable() {
		return kafkaDebugEnable;
	}

	public void setEnableKafkaDebug(String kafkaDebugEnable) {
		this.kafkaDebugEnable = kafkaDebugEnable;
	}

	public String getKafkaLibsDirs() {
		return kafkaLibsDirs;
	}

	public void setKafkaLibsDirs(String kafkaLibsDirs) {
		this.kafkaLibsDirs = kafkaLibsDirs;
	}

	public String getKafkaVersion() {
		return kafkaVersion;
	}

	public void setKafkaVersion(String kafkaVersion) {
		this.kafkaVersion = kafkaVersion;
	}

	public String getKafkaClusterId() {
		return kafkaClusterId;
	}

	public void setKafkaClusterId(String kafkaClusterId) {
		this.kafkaClusterId = kafkaClusterId;
	}

	public String getDebugQueueName() {
		return debugQueueName;
	}

	public void setDebugQueueName(String debugQueueName) {
		this.debugQueueName = debugQueueName;
	};
}

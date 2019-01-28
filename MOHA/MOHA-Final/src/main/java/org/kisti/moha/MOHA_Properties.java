package org.kisti.moha;

public interface MOHA_Properties {

	public static final String JDL = "moha.jdl";
	public static final String DB = "jdbc:mysql://150.183.250.139/moha";
	public static final String DB_EXECUTOR_TABLE_NAME = "exe";
	public static final String DB_APP_TABLE_NAME = "app";
	public static final String KAFKA_LOG_DIR = "kafka-log";
	public static final String KAFKA_SERVER_PROPERTIES = "broker.properties";
	public static final String CONF_KAKFA_VERSION = "KAKFA_VERSION";
	public static final String CONF_KAFKA_CLUSTER_ID = "KAFKA_CLUSTER_ID";
	public static final String CONF_KAFKA_DEBUG_QUEUE_NAME = "KAFKA_DEBUG_QUEUE_NAME";
	public static final String CONF_KAFKA_DEBUG_ENABLE = "KAFKA_DEBUG_ENABLE";
	public static final String CONF_MYSQL_DEBUG_ENABLE = "MYSQL_DEBUG_ENABLE";
	public static final String CONF_ZOOKEEPER_BOOTSTRAP_SERVER = "ZOOKEEPER_BOOTSTRAP_SERVER";
	public static final String CONF_ZOOKEEPER_CONNECT = "ZOOKEEPER_CONNECT";
	public static final String CONF_ZOOKEEPER_SERVER = "ZOOKEEPER_SERVER";
	public static final String CONF_QUEUE_TYPE = "QUEUE_TYPE";
	public static final String CONF_ACTIVEMQ_SERVER = "ACTIVEMQ_SERVER";
	public static final String KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER = "KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER";
	public static final String KAFKA_ZOOKEEPER_CONNECT = "KAFKA_ZOOKEEPER_CONNECT";
	public static final String APP_JAR = "APP_JAR";
	public static final String APP_JAR_TIMESTAMP = "APP_JAR_TIMESTAMP";
	public static final String APP_JAR_SIZE = "APP_JAR_SIZE";
	public static final String KAFKA_TGZ = "KAFKA_TGZ";
	public static final String KAFKA_TGZ_TIMESTAMP = "KAFKA_TGZ_TIMESTAMP";
	public static final String KAFKA_TGZ_SIZE = "KAFKA_TGZ_SIZE";

	public static final String MOHA_TGZ = "KAFKA_TGZ";
	public static final String MOHA_TGZ_TIMESTAMP = "KAFKA_TGZ_TIMESTAMP";
	public static final String MOHA_TGZ_SIZE = "KAFKA_TGZ_SIZE";
	
	
	public static final String ZOOKEEPER_ROOT = "moha_kafka";
	//public static final String ZOOKEEPER_ROOT_MOHA = "kafka_cluster";
	
	public static final String ZOOKEEPER_DIR_REQUEST_STOP = "stop_request";//requests start or stop for Kafka cluster

	public static final String ZOOKEEPER_DIR_LOGS = "logs";	//logs information
	public static final String ZOOKEEPER_DIR_STATUS = "status";//running or not
	public static final String ZOOKEEPER_DIR_RESULTS_APP_LOG = "app_results_log";//To store information of application
	public static final String ZOOKEEPER_DIR_RESULTS_EXE_LOG = "exe_results_log";//To store information of task executors
	public static final String ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG = "exe_performance_log";
	public static final String ZOOKEEPER_DIR_POLLING_ENABLE = "polling_enable";//running or not
	public static final String ZOOKEEPER_DIR_EXECUTORS = "executors";//executors information

	public static final long MOHA_MANAGER_OVERHEAD = 10 * 1000;// Ten seconds
	public static final long KAFKA_BROKERS_TIMEOUT = 5 * 60 * 1000;// Five
																	// minutes
	public static final String EXECUTABLE_DIR = "exe";//executors information
	public static final long TIMMING_INIT = 0;
	public static final long TIMMING_PUSHING = 1;
	public static final long TIMMING_PUSHING_FINISH = 2;
	public static final long TIMMING_FETCHING = 3;
	public static final long TIMMING_FETCHING_FINISH = 4;
	
	public static final String QUEUE_LOGS = "logs";	//logs information
	public static final String ZOOKEEPER_DIR_TIME_START = "time_start";
	public static final String ZOOKEEPER_DIR_TIME_COMPLETE = "time_complete";
	public static final String ZOOKEEPER_DIR_NUM_PROCESSED_TASKS = "num_processed_tasks";
	public static final long SESSION_MAXTIME_TIMEOUT = 24*60*60*1000;
	public static final String HDFS_HOME_DIRECTORY = "hdfs_home_directory";
	public static final String APP_TYPE = "APP_TYPE";
	public static final String ZOOKEEPER_DIR_LOCAL_RESOURCE = "local_resource";
	
}
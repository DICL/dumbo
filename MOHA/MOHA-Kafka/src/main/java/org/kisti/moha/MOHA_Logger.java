package org.kisti.moha;

import java.util.ArrayList;
import java.util.List;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MOHA_Logger {
	private Logger LOG;
	private MOHA_Queue kafkaDebugQueue;
	private MOHA_Queue logQueue;
	private boolean isKafkaAvailable;
	private boolean isLogEnable;
	private boolean isRegistered = false;
	private boolean isSubscribed = false;

	public MOHA_Logger(Class<?> clazz, boolean isLogEnable, String debugQueueName, String zkConnect, String zkBootStrapServer, String comQueueName) {
		this(clazz, isLogEnable, debugQueueName);
		logQueue = new MOHA_Queue(zkConnect, zkBootStrapServer, comQueueName + "logs");
	}

	public MOHA_Logger(Class<?> clazz, boolean isLogEnable, String kafkaDebugLogs) {
		LOG = LoggerFactory.getLogger(clazz);
		this.isKafkaAvailable = new MOHA_Zookeeper(null).isKafkaDebugServiceAvailable();
		this.isLogEnable = isLogEnable;

		if (isLogEnable && isKafkaAvailable) {
			kafkaDebugQueue = new MOHA_Queue(System.getenv().get(MOHA_Properties.CONF_ZOOKEEPER_CONNECT),
					System.getenv().get(MOHA_Properties.CONF_ZOOKEEPER_BOOTSTRAP_SERVER), kafkaDebugLogs);
			kafkaDebugQueue.register();
		}
		LOG.info(this.toString());
	}

	public void delete() {
		if (isLogEnable && isKafkaAvailable) {
			kafkaDebugQueue.deleteQueue();
		}

	}

	/*
	 * Create a kafka_topic for delivering information from Task Executors to
	 * Client
	 */
	public void init() {
		logQueue.create(1, 1);
	}
	public void close(){
		logQueue.deleteQueue();
	}

	/* This function must be called before logging information */
	public void register() {
		logQueue.register();
		isRegistered = true;
	}

	/* This function must be called before getting information */
	public void subcribe() {
		logQueue.subcribe();
		isSubscribed = true;
	}

	/* Send messages to the log queue */
	private void push(String msg) {
		if(isRegistered){
			logQueue.push(msg);
		}
		
	}

	/* Get messages from log queue */
	public List<String> getLogs() {
		if(!isSubscribed)return null;
		List<String> msgs = new ArrayList<String>();
		ConsumerRecords<String, String> records = logQueue.poll(100);
		for (ConsumerRecord<String, String> record : records) {
			msgs.add(record.value());
		}
		return msgs;
	}

	@Override
	public String toString() {
		return "MOHA_Logger [debugQueue=" + kafkaDebugQueue + ", isKafkaAvailable=" + isKafkaAvailable + ", isLogEnable=" + isLogEnable + "]";
	}

	public void queue(String msg) {
		if (isLogEnable && isKafkaAvailable) {
			kafkaDebugQueue.push(msg);
		}
	}

	public void info(String msg) {
		LOG.info(msg);
	}

	public void connect(String msg) {
		// zkconnect.setLogs(msg);
	}

	public void debug(String msg) {
		info(msg);
		queue(msg);
	}

	public void inform(String msg) {

	}

	public void all(String msg) {
		info(msg);
		queue(msg);
		// connect(msg);
		push(msg);
	}

}

package org.kisti.moha;

import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import org.I0Itec.zkclient.ZkClient;
import org.I0Itec.zkclient.ZkConnection;
import org.apache.kafka.clients.consumer.ConsumerConfig;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.clients.consumer.KafkaConsumer;
import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.Producer;
import org.apache.kafka.clients.producer.ProducerRecord;
import org.apache.kafka.common.PartitionInfo;
import org.apache.kafka.common.TopicPartition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import kafka.admin.AdminUtils;
import kafka.utils.ZkUtils;
import kafka.utils.ZkUtils$;

public class MOHA_Queue {

	private static final Logger LOG = LoggerFactory.getLogger(MOHA_Queue.class);

	private final int sessionTimeout = 30000;
	private final int connectionTimeout = 30000;
	private ZkClient zkClient;
	private ZkConnection zkConnection;

	private ZkUtils zkUtils;
	private KafkaConsumer<String, String> consumer;
	private Producer<String, String> producer;

	private String queueName;
	private String bootstrapServers;
	private String zookeeperConnect;

	public MOHA_Queue(String zookeeperConnect, String bootstrapServers, String queueName) {
		this.zookeeperConnect = zookeeperConnect;
		this.bootstrapServers = bootstrapServers;
		this.queueName = queueName;
		LOG.info(this.toString());
	}

	@Override
	public String toString() {
		return "MOHA_Queue [sessionTimeout=" + sessionTimeout + ", connectionTimeout=" + connectionTimeout
				+ ", zkClient=" + zkClient + ", zkConnection=" + zkConnection + ", zkUtils=" + zkUtils + ", consumer="
				+ consumer + ", producer=" + producer + ", queueName=" + queueName + ", bootstrapServers="
				+ bootstrapServers + ", zookeeperConnect=" + zookeeperConnect + "]";
	}

	// Create queue
	public boolean create(int numPartitions, int numReplicationFactor) {

		zkClient = ZkUtils$.MODULE$.createZkClient(zookeeperConnect, sessionTimeout, connectionTimeout);
		zkConnection = new ZkConnection(zookeeperConnect, sessionTimeout);
		zkUtils = new ZkUtils(zkClient, zkConnection, false);
		//for 0.10.1.0
		AdminUtils.createTopic(zkUtils, queueName, numPartitions, numReplicationFactor, new Properties(), null);
		
		
		//AdminUtils.createTopic(zkUtils, queueName, numPartitions, numReplicationFactor, new Properties());

		return true;

	}

	// Delete queue
	public boolean deleteQueue() {
		AdminUtils.deleteTopic(zkUtils, queueName);
		return true;
	}

	// Register to push messages to queue
	public boolean register() {
		
		/*
		LOG.info("bootstramservers = {}", bootstrapServers);
		Properties props = new Properties();
		props.put("bootstrap.servers", bootstrapServers);

		props.put("acks", "all");
		props.put("retries", 0);
		props.put("batch.size", 16384);
		props.put("linger.ms", 1);
		props.put("buffer.memory", 33554432);
		props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
*/
		Properties props = new Properties();
		 props.put("bootstrap.servers", bootstrapServers);
		 props.put("acks", "all");
		 props.put("retries", 0);
		 props.put("batch.size", 16384);
		 props.put("linger.ms", 1);
		 props.put("buffer.memory", 33554432);
		 props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		 props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
		 
		producer = new KafkaProducer<>(props);
		return true;
	}

	public boolean unregister() {
		producer.close();

		return true;
	}

	public boolean commitSync() {
		consumer.commitSync();
		return true;
	}

	// Subscribe to poll messages from queue
	public boolean subcribe() {
		//Properties props = new Properties();
		// int fetch_size = 64;

		// props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
		// props.put(ConsumerConfig.GROUP_ID_CONFIG, queueName);
		// props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
		// props.put(ConsumerConfig.AUTO_COMMIT_INTERVAL_MS_CONFIG, "1000");
		// props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, "30000");
		// props.put("request.timeout.ms", "40000");
		//
		// //props.put("max.partition.fetch.bytes", String.valueOf(fetch_size));
		// props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
		// props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG,
		// "org.apache.kafka.common.serialization.IntegerDeserializer");
		// props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG,
		// "org.apache.kafka.common.serialization.StringDeserializer");
/*(
		props.put("bootstrap.servers", bootstrapServers);
		
		//for testing
		props.put("group.id", queueName); 
		props.put("enable.auto.commit", "false");
		props.put("auto.commit.interval.ms", "1000");
		props.put("session.timeout.ms", "10000");
		props.put("request.timeout.ms", "40000");
		//props.put("max.partition.fetch.bytes", String.valueOf(2*1024));
		//props.put("max.poll.records", "1000000");
		props.put("auto.offset.reset", "earliest");
		props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
		props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
*/
		Properties props = new Properties();
//	    props.put("bootstrap.servers", bootstrapServers);
//	    props.put("group.id", queueName);
//	    props.put("enable.auto.commit", "false");
//	    props.put("auto.commit.interval.ms", "1000");
//	    props.put("auto.offset.reset", "earliest");
//	    
//	    props.put("max.partition.fetch.bytes", String.valueOf(100*1024));
//	    
//	    props.put("session.timeout.ms", "30000");
//	    props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
//	    props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
		
		props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, bootstrapServers);
		props.put("request.timeout.ms", "400000");
		props.put("max.poll.records", "1");
		props.put(ConsumerConfig.AUTO_OFFSET_RESET_CONFIG, "earliest");
		props.put(ConsumerConfig.GROUP_ID_CONFIG, queueName);
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, "false");
        props.put(ConsumerConfig.AUTO_COMMIT_INTERVAL_MS_CONFIG, "1000");
        props.put(ConsumerConfig.SESSION_TIMEOUT_MS_CONFIG, "300000");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer");
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer");
		
		
	    consumer = new KafkaConsumer<>(props);
	    
	    consumer.subscribe(Collections.singletonList(queueName));
	    
	     LOG.info("queue name = {}",queueName);
		return true;
	}

	public boolean close() {
		consumer.close();

		return true;
	}

	public String push(String key, String messages) {
		producer.send(new ProducerRecord<String, String>(queueName, key, messages));		
		return messages;
	}
	
	public List<PartitionInfo> partitionsFor(){
		return producer.partitionsFor(queueName);
	}

	public String push(String messages) {
		producer.send(new ProducerRecord<String, String>(queueName, "0", messages));
		return messages;
	}

	public ConsumerRecords<String, String> poll(int timeOut) {		
		ConsumerRecords<String, String> records = consumer.poll(timeOut);		
		return records;
	}
	public Set<TopicPartition> assignment(){
		return consumer.assignment();
	}
	public List<PartitionInfo> cPartitionsFor(){
		return consumer.partitionsFor(queueName);
	}
	
}

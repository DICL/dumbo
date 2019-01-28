package org.kisti.moha;

import java.util.Collections;
import java.util.List;
import java.util.Properties;
import java.util.Set;

import javax.jms.Connection;
import javax.jms.DeliveryMode;
import javax.jms.Destination;
import javax.jms.ExceptionListener;
import javax.jms.JMSException;
import javax.jms.Message;
import javax.jms.MessageConsumer;
import javax.jms.MessageProducer;
import javax.jms.Session;
import javax.jms.TextMessage;

import org.I0Itec.zkclient.ZkClient;
import org.I0Itec.zkclient.ZkConnection;
import org.apache.activemq.ActiveMQConnectionFactory;
import org.apache.activemq.command.ActiveMQQueue;
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
	private KafkaConsumer<String, String> kafkaConsumer;
	private Producer<String, String> kafkaProducer;

	private String queueName;
	private String bootstrapServers;
	private String zookeeperConnect;

	private String activeMQHost;
	Destination activeMQDestination;
	Session activeMQSession;
	Connection activeMQConnection;
	MessageProducer activeMQProducer;
	MessageConsumer activeMQConsumer;
	Message activeMQMessage;
	// MessageConsumer consumer;

	boolean isKafka = false;
	@Override
	public String toString() {
		return "MOHA_Queue [sessionTimeout=" + sessionTimeout + ", connectionTimeout=" + connectionTimeout + ", zkClient=" + zkClient + ", zkConnection=" + zkConnection + ", zkUtils=" + zkUtils
				+ ", kafkaConsumer=" + kafkaConsumer + ", kafkaProducer=" + kafkaProducer + ", queueName=" + queueName + ", bootstrapServers=" + bootstrapServers + ", zookeeperConnect="
				+ zookeeperConnect + ", activeMQHost=" + activeMQHost + ", activeMQDestination=" + activeMQDestination + ", activeMQSession=" + activeMQSession + ", activeMQConnection="
				+ activeMQConnection + ", activeMQProducer=" + activeMQProducer + ", activeMQConsumer=" + activeMQConsumer + ", activeMQMessage=" + activeMQMessage + ", isKafka=" + isKafka
				+ ", isActiveMQ=" + isActiveMQ + "]";
	}

	boolean isActiveMQ = false;

	public MOHA_Queue(String host, String queueName) {
		this.activeMQHost = host;
		this.queueName = queueName;
		this.isActiveMQ = true;
		LOG.info(this.toString());
	}

	public MOHA_Queue(String zookeeperConnect, String bootstrapServers, String queueName) {
		this.zookeeperConnect = zookeeperConnect;
		this.bootstrapServers = bootstrapServers;
		this.queueName = queueName;
		this.isKafka = true;
		LOG.info(this.toString());
	}

	public Message activeMQPoll(int timeout) {
		try {

			// Wait for a message
			activeMQMessage = activeMQConsumer.receive(timeout);
			//System.out.println("message:  " + message);
			return activeMQMessage;
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}

	// Create queue
	public boolean queueCreate(int numPartitions, int numReplicationFactor) {
		if (isKafka) {
			zkClient = ZkUtils$.MODULE$.createZkClient(zookeeperConnect, sessionTimeout, connectionTimeout);
			zkConnection = new ZkConnection(zookeeperConnect, sessionTimeout);
			zkUtils = new ZkUtils(zkClient, zkConnection, false);
			// for 0.10.1.0
			AdminUtils.createTopic(zkUtils, queueName, numPartitions, numReplicationFactor, new Properties(), null);
		}

		return true;

	}

	// Delete queue
	public boolean deleteQueue() {
		if (isKafka) {
			AdminUtils.deleteTopic(zkUtils, queueName);
		}
		return true;
	}

	// Register to push messages to queue
	public boolean producerInit() {

		if (isKafka) {
			Properties props = new Properties();
			props.put("bootstrap.servers", bootstrapServers);

			props.put("acks", "all");
			props.put("retries", 0);
			props.put("batch.size", 16384);
			props.put("linger.ms", 1);
			props.put("buffer.memory", 33554432);
			props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
			props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

			kafkaProducer = new KafkaProducer<>(props);

		} else if (isActiveMQ) {
			// Create a ConnectionFactory
			//ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory("nio://hdp01.kisti.re.kr:61616");
			ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory("nio://" + activeMQHost);

			// Create a Connection

			try {
				activeMQConnection = connectionFactory.createConnection();
				activeMQConnection.start();

				// Create a Session
				activeMQSession = activeMQConnection.createSession(false, Session.AUTO_ACKNOWLEDGE);

				// Create the destination (Topic or Queue)
				activeMQDestination = activeMQSession.createQueue(queueName + "ACTIVE.MQ");

				// Create a MessageProducer from the Session to the Topic or
				// Queue
				activeMQProducer = activeMQSession.createProducer(activeMQDestination);
				activeMQProducer.setDeliveryMode(DeliveryMode.PERSISTENT);
			} catch (JMSException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

			// Create a messages
		}
		return true;
	}

	public boolean commitSync() {
		if (isKafka) {
			kafkaConsumer.commitSync();
		}
		return true;
	}

	// Subscribe to poll messages from queue
	public boolean consumerInit() {

		if (isKafka) {
			Properties props = new Properties();

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

			kafkaConsumer = new KafkaConsumer<>(props);

			kafkaConsumer.subscribe(Collections.singletonList(queueName));
		} else if (isActiveMQ) {
			// Create a ConnectionFactory
			//ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory("nio://hdp01.kisti.re.kr:61616");
			ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory("nio://"+activeMQHost);
			System.out.println("connectionFactory:  " + connectionFactory);
			// Create a Connection
			Connection connection;
			try {
				connection = connectionFactory.createConnection();
				connection.start();

				// connection.setExceptionListener((ExceptionListener) this);

				// Create a Session
				activeMQSession = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
				System.out.println("session:  " + activeMQSession);
				// Create the destination (Topic or Queue)
				//Destination destination = activeMQSession.createQueue(queueName + "ACTIVE.MQ");
				//System.out.println("destination:  " + destination);
				
				Destination destination = new ActiveMQQueue(queueName + "ACTIVE.MQ" + "?"+"consumer.prefetchSize=1");
				
				// Create a MessageConsumer from the Session to the Topic or Queue
				activeMQConsumer = activeMQSession.createConsumer(destination);
				System.out.println("consumer:  " + activeMQConsumer);
			} catch (JMSException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		LOG.info("queue name = {}", queueName);
		return true;
	}

	public boolean close() {
		if (isKafka) {
			if (kafkaConsumer != null) {
				kafkaConsumer.close();
			}
			if (kafkaProducer != null) {
				kafkaProducer.close();
			}

		} else if (isActiveMQ) {
			try {
				if (activeMQSession != null)
					activeMQSession.close();
				if (activeMQConnection != null)
					activeMQConnection.close();
				if (activeMQConsumer != null)
					activeMQConsumer.close();
			} catch (JMSException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		return true;
	}

	public String push(String messages) {
		if (isKafka) {
			kafkaProducer.send(new ProducerRecord<String, String>(queueName, "0", messages));
		} else if (isActiveMQ) {

			try {

				TextMessage message = activeMQSession.createTextMessage(messages);

				// Tell the producer to send the message
				System.out.println("Sent message: " + message + " : " + Thread.currentThread().getName());
				activeMQProducer.send(message);

				// Clean up

			} catch (JMSException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return messages;
	}

	public String push(String index, String messages) {
		if (isKafka) {
			kafkaProducer.send(new ProducerRecord<String, String>(queueName, index, messages));
		} else if (isActiveMQ) {
			try {
				// Create a messages
				TextMessage message = activeMQSession.createTextMessage(messages);
				activeMQProducer.send(message);
				//System.out.println(messages.toString());
			} catch (JMSException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				System.out.println(e.toString());
			}
		}
		return messages;
	}

	public ConsumerRecords<String, String> poll(int timeOut) {
		if (isKafka) {
			ConsumerRecords<String, String> records = kafkaConsumer.poll(timeOut);
			return records;
		}

		return null;
	}

	public Set<TopicPartition> assignment() {
		if (isKafka) {
			return kafkaConsumer.assignment();
		}
		return null;
	}

	public List<PartitionInfo> partitionsFor() {
		if (isKafka) {
			return kafkaConsumer.partitionsFor(queueName);
		}
		return null;
	}

	public boolean isKafka() {
		return isKafka;
	}

}

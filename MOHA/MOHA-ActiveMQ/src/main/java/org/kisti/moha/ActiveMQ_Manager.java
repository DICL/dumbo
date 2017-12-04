package org.kisti.moha;

import javax.jms.Connection;
import javax.jms.DeliveryMode;
import javax.jms.Destination;
import javax.jms.JMSException;
import javax.jms.Message;
import javax.jms.MessageConsumer;
import javax.jms.MessageProducer;
import javax.jms.Session;
import javax.jms.TextMessage;

import org.apache.activemq.ActiveMQConnectionFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class ActiveMQ_Manager {
	private static final Logger LOG = LoggerFactory.getLogger(ActiveMQ_Manager.class);
	private String QueueName;
	private String QueueLocation;

	/* ActiveMQ Connection Information */
	private ActiveMQConnectionFactory connectionFactory;
	private Connection connection;
	private Session session;
	private Destination destination;
	private MessageProducer producer;
	private MessageConsumer consumer;
	
	
	public ActiveMQ_Manager(String q_name, String q_loc, int AccessType) {
		this.QueueName = q_name;
		this.QueueLocation = q_loc;
		
		/* Initialize ActiveMQ Access */		
        this.connectionFactory = new ActiveMQConnectionFactory(this.QueueLocation);
        
        /* We try to share connection related objects as much as possible */
        try {
			this.connection = this.connectionFactory.createConnection();
			this.connection.start();
			this.session = this.connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
			this.destination = session.createQueue(this.QueueName);
			
			if(AccessType == MOHA_Constants.ACTIVEMQ_PRODUCER) {
				this.producer = session.createProducer(this.destination);
				this.producer.setDeliveryMode(DeliveryMode.NON_PERSISTENT);
			}		
		
			else if(AccessType == MOHA_Constants.ACTIVEMQ_CONSUMER) {
				this.consumer = session.createConsumer(this.destination);
			}
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}            
	}//The end of constructor
	
	
	public boolean SimpleInsertTasks(int num_tasks, String command) {
		int index = 0;
		TextMessage message;
		
		LOG.info("Inserting {} of {} tasks into the ActiveMQ at {}", num_tasks, command, this.QueueLocation);
				
		try {		
			for(index = 0; index < num_tasks; index++) {        		        	
				message = session.createTextMessage(command);
				this.producer.send(message);
			} 
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
			    
		LOG.info("Completed the insertion of tasks !");
    		
		return true;
	}//The end of SimpleInsertTasks function
	
	
	public String SimpleRetrieveTask() {
		TextMessage textMsg;
        String task = null;
        Message message;
        int MessageWaitTime = 500; //originally 1000, smaller is better for scalability
        
		try {
			//Wait for a message
            message = this.consumer.receive(MessageWaitTime);
            
            if (message instanceof TextMessage) {
            	textMsg = (TextMessage) message;
                task = textMsg.getText();
            }   
            
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}      
        
        return task;
	}//The end of SimpleRetrieveTask function

	
	public void Finish_AMQ(int AccessType) {		
		try {
			if(AccessType == MOHA_Constants.ACTIVEMQ_PRODUCER) {
				this.producer.close();
			}
				
			else if(AccessType == MOHA_Constants.ACTIVEMQ_CONSUMER) {
				this.consumer.close();	
			}
				
			this.session.close();
			this.connection.close();
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		 
	}//The end of Finish_AMQ function
	
	
	
	/***************************************** DEPRECATED FUNCTIONS *****************************************/
	/*
	 * The following functions (InsertTasks, RetrieveTask) may work without any problems in terms of 
	 * inserting tasks into the ActiveMQ and retrieving tasks from the queue.
	 * However, due to frequent creation and closure of sessions and connections, they show relatively
	 * poor task dispatching and insertion performance. Therefore, we implemented "Simple" version of
	 * insertion and retrieval functionalities which can share the session and connection objects
	 * as much as possible in this class's instances.   
	 */
	public boolean InsertTasks(int num_tasks, String command) {
		int index = 0;
		
		LOG.info("Inserting {} of {} tasks into the ActiveMQ at {}", num_tasks, command, this.QueueLocation);
		//Create a ConnectionFactory
        ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory(this.QueueLocation);
        
        Connection connection;
		try {
	        //Create a Connection
			connection = connectionFactory.createConnection();
			connection.start();
			
			//Create a Session
	        Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);
	        
	        //Create the destination (Topic or Queue)
	        Destination destination = session.createQueue(this.QueueName);
	        
	        //Create a MessageProducer from the Session to the Topic or Queue
	        MessageProducer producer = session.createProducer(destination);
	        producer.setDeliveryMode(DeliveryMode.NON_PERSISTENT);

	        for(index = 0; index < num_tasks; index++) {        		        	
	        	TextMessage message = session.createTextMessage(command);
	        	producer.send(message);
	        }
	        
	        //Clean up
	        producer.close();
	        session.close();
	        connection.close();	
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return false;
		}
		
		LOG.info("Completed the insertion of tasks !");
    		
		return true;
	}//The end of InsertTasks function
	
	
	public String RetrieveTask() {
		//Create a ConnectionFactory
        ActiveMQConnectionFactory connectionFactory = new ActiveMQConnectionFactory(this.QueueLocation);
        TextMessage textMsg;
        String task = null;
        int MessageWaitTime = 500; //originally 1000
        
        Connection connection;
		try {
			//Create a Connection
			connection = connectionFactory.createConnection();
			connection.start();
			
			//Create a Session
            Session session = connection.createSession(false, Session.AUTO_ACKNOWLEDGE);

            //Create the destination (Topic or Queue)
            Destination destination = session.createQueue(this.QueueName);

            //Create a MessageConsumer from the Session to the Topic or Queue
            MessageConsumer consumer = session.createConsumer(destination);

            //Wait for a message
            Message message = consumer.receive(MessageWaitTime);
            
            if (message instanceof TextMessage) {
            	textMsg = (TextMessage) message;
                task = textMsg.getText();
            }    
            
            consumer.close();
            session.close();
            connection.close();
		} catch (JMSException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			return null;
		}      
        
        return task;
	}//The end of RetrieveTasks function
		
}//The end of ActiveMQ_Manager class

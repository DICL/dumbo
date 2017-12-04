package org.kisti.moha;

public interface MOHA_Constants {
	
	public static final String APP_NAME = "MOHA"; //Fix the name of YARN application as "MOHA"
	
	public static final String APP_JAR_NAME = "MOHA.jar"; //Fix the name of Jar file to be stored in HDFS
	
	public static final String JDL_NAME = "INPUT.jdl"; //Fix the name of JDL file to be stored in HDFS
	
	/* The default number of retrying to check whether the queue is empty or not */
	public static final String DEFAULT_EXECUTOR_RETRIALS = "3"; 
		
	public static final int ACTIVEMQ_PRODUCER = 0;
	
	public static final int ACTIVEMQ_CONSUMER = 1;
}

package org.kisti.moha;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.util.Date;
import java.util.Map;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.net.NetUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MOHA_TaskExecutor {
	private static final Logger LOG = LoggerFactory.getLogger(MOHA_TaskExecutor.class);
	private String ActiveMQ_URL = "";
	private String ApplicationID = "";
	private int NumOfRetrialsToQueue = 0;	
	private static int NumOfExecutedTasks = 0;
	
	
	public MOHA_TaskExecutor(String[] args) throws IOException {
		Map<String, String> env = System.getenv();
		this.ActiveMQ_URL = env.get("ACTIVEMQ_URL");		
		LOG.info("The location of ActiveMQ Service is {}", this.ActiveMQ_URL);
		
		this.ApplicationID = System.getenv().get("MOHA_APP_ID");
		LOG.info("The Application ID of MOHA_TaskExecutor is {}", this.ApplicationID);
		
		this.NumOfRetrialsToQueue = Integer.parseInt(System.getenv().get("EXECUTOR_RETRIALS"));
		LOG.info("The number of retrials to access the job queue is {}", this.NumOfRetrialsToQueue);
	}//The end of constructor
	
	
	public static void main(String[] args) {
		long startTime, finishTime;
		double elapsedTime, performance;
		
		LOG.info("MOHA_TaskExecutor just started on {}", NetUtils.getHostname());
		startTime = new Date().getTime();		
		
		try {
			MOHA_TaskExecutor container = new MOHA_TaskExecutor(args);
			container.run();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		finishTime = new Date().getTime();
		LOG.info("MOHA_TaskExecutor completes !");
		
		elapsedTime = (double)(finishTime - startTime)/(double)1000;
		performance = (double)NumOfExecutedTasks / (double)elapsedTime;
		
		LOG.info("This MOHA_TaskExecutor processed {} tasks in {} (sec): {} tasks/sec", 
				NumOfExecutedTasks, elapsedTime, performance);		
	}//The end of main function

	
	public void run() {		
		ActiveMQ_Manager amq_manager = new ActiveMQ_Manager(ApplicationID, ActiveMQ_URL, MOHA_Constants.ACTIVEMQ_CONSUMER);
		String task;
		
		/* Number of retrials to check whether the ActiveMQ is empty */
		int NumOfRetrials = 0;
					
		while(true) {			
			task = amq_manager.SimpleRetrieveTask();
			
			//We should determine whether the MOHA Job Queue is empty or just busy
			if(StringUtils.isEmpty(task)) {				
				NumOfRetrials++;
				if(NumOfRetrials <= this.NumOfRetrialsToQueue) {
					try {
						Thread.sleep(1000);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					continue;
				}
				
				break;
			}
			
			/* Execute the command */
			//run_shell_by_processbuilder(task);
			//run_shell_by_runtime(task);
			
			NumOfExecutedTasks++;
		}//The end of while				
		
		amq_manager.Finish_AMQ(MOHA_Constants.ACTIVEMQ_CONSUMER);
	}//The end of run function
	

	private void run_shell_by_runtime(String command) {
		//Executing the shell command		
		Runtime run_command = Runtime.getRuntime();
		Process pr = null;
		try {
			pr = run_command.exec(command);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		try {
			pr.waitFor();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		BufferedReader buf = new BufferedReader(new InputStreamReader(pr.getInputStream()));
		String line = "";
		try {
			while ((line=buf.readLine())!=null) {
				System.out.println(line);
			}
			buf.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}			
	}//The end of run_shell_by_runtime function
	
	
	private void run_shell_by_processbuilder(String command) {	        
	    /*
	     * The ProcessBuilder should take an array of strings especially when
	     * there are whitespaces in the command
	    */
	    String[] command_array = command.split("\\s+");
	    ProcessBuilder builder = new ProcessBuilder(command_array);
	    Process pr = null;
	    		
	    try {
			pr = builder.start();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	    
	    
	    BufferedReader buf = new BufferedReader(new InputStreamReader(pr.getInputStream()));
		String line = "";
		try {
			while ((line=buf.readLine())!=null) {
				System.out.println(line);
			}
			buf.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}	
	}//The end of run_shell_by_processbuilder function		
	
}//The end of MOHA_TaskExecutor class

package org.kisti.moha;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Vector;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.GetNewApplicationResponse;
import org.apache.hadoop.yarn.api.records.ApplicationId;
import org.apache.hadoop.yarn.api.records.ApplicationSubmissionContext;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.NodeState;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.QueueInfo;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.api.records.YarnClusterMetrics;
import org.apache.hadoop.yarn.client.api.YarnClient;
import org.apache.hadoop.yarn.client.api.YarnClientApplication;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MOHA_Client {

	private static final Logger LOG = LoggerFactory.getLogger(MOHA_Client.class);
	
	private YarnConfiguration conf;
	private YarnClient yarnClient;	
	private ApplicationId appId;
	private FileSystem fs;
	private Options opts;
	
	//List of command-line options		
	private int mohaPriority = 0;
	private String managerQueue = "default";
	private int managerMemory = 1024;	
	private String appJar = MOHA_Constants.APP_JAR_NAME;
	private int executorMemory = 1024;
	private int numExecutors = 1;
	private String JDLFile = "";
		
	private String ActiveMQ_URL = null; //The location of ActiveMQ Service
	private String ActiveMQ_Home = null; //The location of ActiveMQ installation
	private String TaskExecutor_Retrials = null; //The default value for MOHA_TaskExecutor's retrials to access the job queue is 3
	
	public static void main(String[] args) {
		MOHA_Client client;
		boolean result = false;
		
		LOG.info("Initializing the MOHA_Client");

		try {
			client = new MOHA_Client(args);
			result = client.init(args);
			
			if(!result) {
				LOG.error("Finishing the MOHA execution without YARN submission ...");
				return;
			}
			
			result = client.run();
		} catch (IOException | ParseException | YarnException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}			
		
		if(result) {
			LOG.info("The MOHA_Client is successfully executed");
		}
	}//The end of main function

	
	public MOHA_Client(String[] args) throws IOException {
		conf = new YarnConfiguration();
		yarnClient = YarnClient.createYarnClient();
		yarnClient.init(conf);
		fs = FileSystem.get(conf);
		
		opts = new Options();
		
		/*
		 * Add an option that only contains a short-name. It may be specified as requiring an argument.
		   - Parameters
		     : opt (Short single-character name of the option)
		     : hasArg flag (signally if an argument is required after this option)
		     : description (Self-documenting description)
		   - Returns: the resulting Options instance
		 */
		opts.addOption("priority", true, "Application Priority (Default: 0)");
		opts.addOption("queue", true, 
				"RM Queue where MOHA_Manager is submitted (Default: default)");
		opts.addOption("manager_memory", true, 
				"Amount of memory in MB for MOHA Manager (Default: 1024)");
		opts.addOption("jar", true, 
				"JAR file containing the MOHA Manager and Task Executor (Default: MOHA.jar)");
		opts.addOption("executor_memory", true, 
				"Amount of memory in MB for MOHA TaskExecutor (Default: 1024)");
				
		opts.addOption("num_executors", true, "Number of MOHA Task Executors (Default: 1)");
		opts.addOption("JDL", true, "Job Description Language file");
		opts.addOption("help", false, "Print Usage of MOHA_Client");		
	}//The end of constructor
	
	
	private void printUsage() {
		new HelpFormatter().printHelp("MOHA_Client", opts);
	}
	
	
	public boolean init(String[] args) throws ParseException {
		CommandLine cliParser = new GnuParser().parse(opts, args);
		
		if (cliParser.hasOption("help")) {
			printUsage();
			return false;
		}
							
		/* Parse the input arguments */
		mohaPriority = Integer.parseInt(cliParser.getOptionValue("priority", "0"));
		LOG.info("Priority of MOHA_Manager & TaskExecutor = {}", mohaPriority);
		
		managerQueue = cliParser.getOptionValue("queue", "default");
		LOG.info("Destination queue of MOHA_Manager = {}", managerQueue);
		
		managerMemory = Integer.parseInt(cliParser.getOptionValue("manager_memory", "1024"));
		LOG.info("Memory amount of MOHA_Manager = {}", managerMemory);
		
		appJar = cliParser.getOptionValue("jar", "MOHA.jar");
		LOG.info("JAR file containing the MOHA Application = {}", appJar);
		
		executorMemory = Integer.parseInt(cliParser.getOptionValue("executor_memory", "1024"));
		LOG.info("Amount of memory in MB for MOHA TaskExecutor = {}", executorMemory);
		
		numExecutors = Integer.parseInt(cliParser.getOptionValue("num_executors", "1"));	
		LOG.info("Number of MOHA Task Executors = {}", numExecutors);
		
		//The Job Description File is necessary to execute MOHA tasks
		if(!cliParser.hasOption("JDL")) {
			LOG.error("The Job Description File should be provided !");
			return false;
		}
		JDLFile = cliParser.getOptionValue("JDL");
		LOG.info("The Job Description File = {}", JDLFile);
		
		/* Parse the MOHA configuration file */
		Properties prop = new Properties();
		
		try {
			//The MOHA configuration file is always static
			prop.load(new FileInputStream("conf/MOHA.conf"));
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			LOG.error("MOHA configuration file cannot be found: conf/MOHA.conf");
			return false;
			//return false;
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			LOG.error("MOHA configuration file cannot be found: conf/MOHA.conf");
			return false;
		}
		
		ActiveMQ_URL = prop.getProperty("ActiveMQ.Service");
		if(StringUtils.isEmpty(this.ActiveMQ_URL)) {
			LOG.error("The location of ActiveMQ Service is not specified (MOHA.conf) !");
			return false;
		}
		else {
			LOG.info("The location of ActiveMQ Service = {}", ActiveMQ_URL);
		}
		
		this.ActiveMQ_Home = prop.getProperty("ActiveMQ.Home");
		if(StringUtils.isEmpty(this.ActiveMQ_Home)) {
			LOG.error("The location of ActiveMQ installation is not specified (MOHA.conf) !");
			return false;
		}
		else {
			LOG.info("The location of ActiveMQ installation = {}", this.ActiveMQ_Home);
		}
		
		this.TaskExecutor_Retrials = prop.getProperty("TaskExecutor.Retrials");
		if(StringUtils.isEmpty(this.TaskExecutor_Retrials)) {
			TaskExecutor_Retrials = MOHA_Constants.DEFAULT_EXECUTOR_RETRIALS;						
		}		
		LOG.info("The number of retrials from TaskExecutor to access the job queue = {}", TaskExecutor_Retrials);
		
		return true;
	}//The end of init function
	

	public boolean run() throws YarnException, IOException {
		yarnClient.start();
		YarnClientApplication client = yarnClient.createApplication();
		GetNewApplicationResponse appResponse = client.getNewApplicationResponse();
		
		//Get a new Application ID assigned by the ResourceManager
		appId = appResponse.getApplicationId();
		LOG.info("Assigned MOHA Applicatoin ID = {}", appId);
		
		/* Get the YARN cluster information */
		int maxMemory =	appResponse.getMaximumResourceCapability().getMemory();
		int maxVCores =	appResponse.getMaximumResourceCapability().getVirtualCores();
		LOG.info("Max Memory = {} and Max Vcores = {}", maxMemory, maxVCores);
		
		YarnClusterMetrics clusterMetrics =
				yarnClient.getYarnClusterMetrics();
		LOG.info("Number of NodeManagers = {}", clusterMetrics.getNumNodeManagers());

		List<NodeReport> nodeReports = yarnClient.getNodeReports(NodeState.RUNNING);
		for (NodeReport node : nodeReports) {
			LOG.info("Node ID = {}, Address = {}, Containers = {}", node.getNodeId(), node.getHttpAddress(),
					node.getNumContainers());
		}
		List<QueueInfo> queueList = yarnClient.getAllQueues();
		for (QueueInfo queue : queueList) {
			LOG.info("Available Queue: {} with capacity {} to {}", queue.getQueueName(), queue.getCapacity(),
					queue.getMaximumCapacity());
		}
		
		/* Prepare the ContainerLaunchContext for the MOHA_Manager */
		ApplicationSubmissionContext appContext = client.getApplicationSubmissionContext();
		appContext.setApplicationName(MOHA_Constants.APP_NAME); //using the fixed application name as "MOHA"
		Map<String, LocalResource> localResources = new HashMap<>();
		
		//Copy the JAR file containing the MOHA Manager and Task Executor into the HDFS
		Path src_jar = new Path(this.appJar);
		String pathSuffix_jar = MOHA_Constants.APP_NAME + "/" + appId.getId() + "/" + MOHA_Constants.APP_JAR_NAME;
		Path dest_jar = new Path(fs.getHomeDirectory(), pathSuffix_jar);
		fs.copyFromLocalFile(false, true, src_jar, dest_jar);
		FileStatus destStatus_jar = fs.getFileStatus(dest_jar);
		
		//Set the MOHA.jar as LocalResource
		LocalResource jarResource = Records.newRecord(LocalResource.class);
		jarResource.setResource(ConverterUtils.getYarnUrlFromPath(dest_jar));
		jarResource.setSize(destStatus_jar.getLen());
		jarResource.setTimestamp(destStatus_jar.getModificationTime());
		jarResource.setType(LocalResourceType.FILE);
		jarResource.setVisibility(LocalResourceVisibility.APPLICATION);		
		localResources.put(MOHA_Constants.APP_JAR_NAME, jarResource);
		
		//Copy the JDL file into the HDFS
		Path src_jdl = new Path(this.JDLFile);
		String pathSuffix_jdl = MOHA_Constants.APP_NAME + "/" + appId.getId() + "/" + MOHA_Constants.JDL_NAME;
		Path dest_jdl = new Path(fs.getHomeDirectory(), pathSuffix_jdl);
		fs.copyFromLocalFile(false, true, src_jdl, dest_jdl);
		FileStatus destStatus_jdl = fs.getFileStatus(dest_jdl);
		
		//Set the JDL file as LocalResource
		LocalResource jdlResource = Records.newRecord(LocalResource.class);
		jdlResource.setResource(ConverterUtils.getYarnUrlFromPath(dest_jdl));
		jdlResource.setSize(destStatus_jdl.getLen());
		jdlResource.setTimestamp(destStatus_jdl.getModificationTime());
		jdlResource.setType(LocalResourceType.FILE);
		jdlResource.setVisibility(LocalResourceVisibility.APPLICATION);		
		localResources.put(MOHA_Constants.JDL_NAME, jdlResource);
		
		//Set the MOHA_Manager's environment
		Map<String, String> env = new HashMap<>();
		String appJarDest = dest_jar.toUri().toString();
		env.put("MOHA_JAR", appJarDest); //Keep the URI of Jar file in the HDFS as an environment variable		
		env.put("MOHA_JAR_TIMESTAMP", Long.toString(destStatus_jar.getModificationTime()));
		env.put("MOHA_JAR_LEN", Long.toString(destStatus_jar.getLen()));
		env.put("MOHA_APP_ID", appId.toString());
		env.put("ACTIVEMQ_URL", this.ActiveMQ_URL); //Keep the location of ActiveMQ Service as an environment variable
		env.put("EXECUTOR_RETRIALS", this.TaskExecutor_Retrials); //Keep the number of retrials from TaskExecutor to the job queue
		
		//Set the class path for running the MOHA
		StringBuilder classPathEnv 
			= new StringBuilder().append(File.pathSeparatorChar).append("./" + MOHA_Constants.APP_JAR_NAME);
		for (String c :
			conf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH, 
					YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
			classPathEnv.append(File.pathSeparatorChar);
			classPathEnv.append(c.trim());
		}
		/* Add ActiveMQ Class Path */
		classPathEnv.append(File.pathSeparatorChar);
		classPathEnv.append(this.ActiveMQ_Home + "*");
				
		classPathEnv.append(File.pathSeparatorChar);
		classPathEnv.append(Environment.CLASSPATH.$());
		env.put("CLASSPATH", classPathEnv.toString());
				
		//Set the Application Submission Context
		ContainerLaunchContext amContainer = Records.newRecord(ContainerLaunchContext.class);
		amContainer.setLocalResources(localResources);
		amContainer.setEnvironment(env);

		//Set the execution command for the MOHA_Manager
		Vector<CharSequence> vargs = new Vector<>(30);		
		vargs.add("java");
		vargs.add("org.kisti.moha.MOHA_Manager");
		vargs.add("--priority " + String.valueOf(mohaPriority));
		vargs.add("--executor_memory " + String.valueOf(executorMemory));
		vargs.add("--num_executors " + String.valueOf(numExecutors));
		vargs.add("1><LOG_DIR>/MOHA_Manager.stdout");
		vargs.add("2><LOG_DIR>/MOHA_Manager.stderr");
		StringBuilder command = new StringBuilder();
		for (CharSequence str : vargs) {
			command.append(str).append(" ");
		}
		List<String> commands = new ArrayList<>();
		commands.add(command.toString());
		LOG.info("Command to execute the MOHA_Manager = {}", command);
		
		amContainer.setCommands(commands);
		
		//Set the capability, priority and queue of MOHA_Manager
		Resource capability = Records.newRecord(Resource.class);
		capability.setMemory(managerMemory);
		appContext.setResource(capability);

		appContext.setAMContainerSpec(amContainer);

		Priority pri = Records.newRecord(Priority.class);
		pri.setPriority(mohaPriority);
		appContext.setPriority(pri);

		appContext.setQueue(managerQueue);
		
		//submit the application (MOHA_Manager)		
		yarnClient.submitApplication(appContext);
		LOG.info("A new MOHA application ({}) is successfully submitted to the YARN", appId.toString());
						
		return true;
	}//The end of run function
		
}//The end of MOHA_Client Class

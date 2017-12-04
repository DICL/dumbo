package org.kisti.moha;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
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
import org.apache.kafka.common.PartitionInfo;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MOHA_Client {
	private static final Logger LOG = LoggerFactory.getLogger(MOHA_Client.class);
	private YarnConfiguration conf;
	private YarnClient yarnClient;
	private ApplicationId appId;
	private FileSystem fs;

	private String appName;
	private int priority;
	private String queue;
	private int managerMemory;
	private String jarPath;
	private int executorMemory;
	private int numExecutors;
	private String jdlPath;
	private long startingTime;

	public static void main(String[] args) throws IOException {
		// [UPDATE] Change the flow of the main function to enable exception
		// handling at each step
		/*
		 * MOHA_Client client = new MOHA_Client(args);
		 * 
		 * try { boolean result = client.run();
		 * LOG.info(String.valueOf(result)); } catch (YarnException e) { // TODO
		 * Auto-generated catch block e.printStackTrace(); } catch (IOException
		 * e) { // TODO Auto-generated catch block e.printStackTrace(); }
		 */
		MOHA_Client client;
		boolean result = false;

		LOG.info("Initializing the MOHA_Client");

		try {
			client = new MOHA_Client(args);
			result = client.init(args);

			if (!result) {
				LOG.info("Finishing the MOHA execution without YARN submission ...");
				return;
			}

			result = client.run();
		} catch (IOException | ParseException | YarnException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		if (result) {
			LOG.info("The MOHA_Client is successfully executed");
		}
	}// The end of main function

	public MOHA_Client(String[] args) throws IOException {
		// [UPDATE] Some logics are shifted into the main function
		/*
		 * try { LOG.info("Start init MOHA_Client"); startingTime =
		 * System.currentTimeMillis(); init(args);
		 * LOG.info("Successfully init"); conf = new YarnConfiguration();
		 * yarnClient = YarnClient.createYarnClient(); yarnClient.init(conf); fs
		 * = FileSystem.get(conf); } catch (ParseException e) { // TODO
		 * Auto-generated catch block e.printStackTrace(); }
		 */
		conf = new YarnConfiguration();
		yarnClient = YarnClient.createYarnClient();
		yarnClient.init(conf);
		fs = FileSystem.get(conf);
	}// The end of MOHA_Client constructor

	public boolean init(String[] args) throws ParseException {
		/*
		 * Add an option that only contains a short-name. It may be specified as
		 * requiring an argument. - Parameters : opt (Short single-character
		 * name of the option) : hasArg flag (signally if an argument is
		 * required after this option) : description (Self-documenting
		 * description) - Returns: the resulting Options instance
		 */
		// [UPDATE] change the hadArg flags into "true" except for the help
		// option
		Options option = new Options();
		option.addOption("appname", true, "MOHA Application Name (Default: MOHA)");
		option.addOption("priority", true, "Application Priority (Default: 0)");
		option.addOption("queue", true, "RM Queue in which this application is to be submitted (Default: default)");
		option.addOption("manager_memory", true, "Amount of memory in MB to be requested to run the MOHA Manager (Default: 1024)");
		option.addOption("jar", true, "JAR file containing the MOHA Manager and Task Executor (Default: MOHA.jar)");
		option.addOption("executor_memory", true, "Amount of memory in MB to be requested to run the MOHA TaskExecutor (Default: 1024)");
		option.addOption("num_executors", true, "Number of MOHA Task Executors (Default: 1)");
		option.addOption("JDL", true, "Job Description Language file that contains the MOHA job specification (must specified)");
		option.addOption("help", false, "Print Usage of MOHA_Client"); // Add
																		// the
																		// help
																		// functionality
																		// in
																		// MOHA_Client

		CommandLine inputParser = new GnuParser().parse(option, args);

		// [UPDATE] Add the help functionality in MOHA_Client
		if (inputParser.hasOption("help")) {
			printUsage(option);
			return false;
		}

		// [UPDATE] Add default values for options
		appName = inputParser.getOptionValue("appname", "MOHA");
		priority = Integer.parseInt(inputParser.getOptionValue("priority", "0"));
		queue = inputParser.getOptionValue("queue", "default");
		managerMemory = Integer.parseInt(inputParser.getOptionValue("manager_memory", "1024"));
		jarPath = inputParser.getOptionValue("jar", "MOHA.jar");
		executorMemory = Integer.parseInt(inputParser.getOptionValue("executor_memory", "1024"));
		numExecutors = Integer.parseInt(inputParser.getOptionValue("num_executors", "1"));

		// [UPDATE] The Job Description File is necessary to execute MOHA tasks
		if (!inputParser.hasOption("JDL")) {
			LOG.error("The Job Description File should be provided !");
			return false;
		}
		jdlPath = inputParser.getOptionValue("JDL");

		// [UPDATE] change the exception handling logic to avoid unnecessary
		// exception throwing
		if (priority < 0) {
			LOG.error("Invalid value is specified for the Application Priority");
			return false;
			// throw new IllegalArgumentException("Invalid value specified for
			// Application Priority");
		}

		// [UPDATE] Unless there is a minimum memory requirement, positive
		// values look O.K.
		// if (managerMemory < 32) {
		if (managerMemory <= 0) {
			LOG.error("Invalid value is specified for the amount of memory of the MOHA Manager");
			return false;
			// throw new IllegalArgumentException(
			// "Invalid value specified for amout of memory in MB to be
			// requested to run the MOHA Manager");
		}

		// if (executorMemory < 32) {
		if (executorMemory <= 0) {
			LOG.error("Invalid value is specified for the amount of memory of the MOHA TaskExecutor");
			return false;
			// throw new IllegalArgumentException(
			// "Invalid value specified for amount of memory in MB to be
			// requested to run the MOHA TaskExecutor");
		}

		if (numExecutors < 1) {
			LOG.error("Invalid value is specified for the number of MOHA TaskExecutors");
			return false;
			// throw new IllegalArgumentException(
			// "Invalid value specified for number of MOHA TaskEcecutor to be
			// executed");
		}

		LOG.info("App name = {}, priority = {}, queue = {}, manager memory = {}, jarPath = {}, executor memory = {}, " + "num ececutors = {}, jdl path = {}",
				appName, priority, queue, managerMemory, jarPath, executorMemory, numExecutors, jdlPath);

		return true;
	}// The end of init function

	private void printUsage(Options opts) {
		new HelpFormatter().printHelp("MOHA_Client", opts);
	}

	public boolean run() throws YarnException, IOException {
		LOG.info("yarnClient = {}", yarnClient.toString());
		yarnClient.start();
		YarnClientApplication yarnClientApplication = yarnClient.createApplication();
		GetNewApplicationResponse appResponse = yarnClientApplication.getNewApplicationResponse();
		appId = appResponse.getApplicationId();

		LOG.info("Application ID = {}", appId);
		int maxMemory = appResponse.getMaximumResourceCapability().getMemory();
		if (managerMemory > (maxMemory)) {
			managerMemory = maxMemory;
		}
		int maxVcores = appResponse.getMaximumResourceCapability().getVirtualCores();
		LOG.info("Max memory = {} and max vcores = {}", maxMemory, maxVcores);
		YarnClusterMetrics clusterMetrics = yarnClient.getYarnClusterMetrics();
		LOG.info("Number of NodeManagers in the Cluster = {}", clusterMetrics.getNumNodeManagers());
		List<NodeReport> nodeReports = yarnClient.getNodeReports(NodeState.RUNNING);
		for (NodeReport node : nodeReports) {
			LOG.info("Node ID = {} , address = {}, container = {}", node.getNodeId(), node.getHttpAddress(), node.getNumContainers());

		}
		List<QueueInfo> nodeQueues = yarnClient.getAllQueues();
		for (QueueInfo queues : nodeQueues) {
			LOG.info("name = {}, capacity = {}, maximum capacity of each queue = {}", queues.getQueueName(), queues.getCapacity(), queues.getMaximumCapacity());
		}
		Path jarSrc = new Path(this.jarPath);
		String rootDir = appName + "/" + appId.getId();
		String jarPathSuffix = rootDir + "/app.jar";
		Path jarDest = new Path(fs.getHomeDirectory(), jarPathSuffix);
		fs.copyFromLocalFile(false, true, jarSrc, jarDest);

		FileStatus jarDestStatus = fs.getFileStatus(jarDest);

		LocalResource jarResource = Records.newRecord(LocalResource.class);
		jarResource.setResource(ConverterUtils.getYarnUrlFromPath(jarDest));
		jarResource.setSize(jarDestStatus.getLen());
		jarResource.setTimestamp(jarDestStatus.getModificationTime());
		jarResource.setType(LocalResourceType.FILE);
		jarResource.setVisibility(LocalResourceVisibility.APPLICATION);
		Map<String, LocalResource> localResources = new HashMap<>();
		localResources.put("app.jar", jarResource);
		LOG.info("Jar resource = {}", jarResource.toString());

		Path mohaSrc = new Path("run.tar.gz");
		String pathSuffixmoha_tgz = appName + "/" + appId.getId() + "/run.tar.gz";
		Path mohaDest = new Path(fs.getHomeDirectory(), pathSuffixmoha_tgz);
		fs.copyFromLocalFile(false, true, mohaSrc, mohaDest);

		FileStatus mohaStatus = fs.getFileLinkStatus(mohaDest);
		LOG.info("FileStatus ={}", mohaStatus.toString());

		MOHA_Configuration mohaConf = new MOHA_Configuration("conf/MOHA.conf");
		LOG.info("Configuration file = {}", mohaConf.toString());

		Map<String, String> env = new HashMap<>();

		env.put(MOHA_Properties.APP_JAR, jarDest.toUri().toString());
		env.put(MOHA_Properties.APP_JAR_TIMESTAMP, Long.toString(jarDestStatus.getModificationTime()));
		env.put(MOHA_Properties.APP_JAR_SIZE, Long.toString(jarDestStatus.getLen()));

		env.put(MOHA_Properties.MOHA_TGZ, mohaDest.toUri().toString());
		env.put(MOHA_Properties.MOHA_TGZ_TIMESTAMP, Long.toString(mohaStatus.getModificationTime()));
		env.put(MOHA_Properties.MOHA_TGZ_SIZE, Long.toString(mohaStatus.getLen()));

		StringBuilder classPathEnv = new StringBuilder().append(File.pathSeparatorChar).append("./app.jar");
		for (String c : conf.getStrings(YarnConfiguration.YARN_APPLICATION_CLASSPATH, YarnConfiguration.DEFAULT_YARN_APPLICATION_CLASSPATH)) {
			classPathEnv.append(File.pathSeparatorChar);
			classPathEnv.append(c.trim());
		}

		classPathEnv.append(File.pathSeparatorChar);
		classPathEnv.append(mohaConf.getKafkaLibsDirs());
		classPathEnv.append(File.pathSeparatorChar);
		classPathEnv.append(Environment.CLASSPATH.$());

		env.put("CLASSPATH", classPathEnv.toString());

		env.put(MOHA_Properties.KAKFA_VERSION, mohaConf.getKafkaVersion());
		env.put(MOHA_Properties.KAFKA_CLUSTER_ID, mohaConf.getKafkaClusterId());
		env.put(MOHA_Properties.KAFKA_DEBUG_QUEUE_NAME, mohaConf.getDebugQueueName());
		env.put(MOHA_Properties.KAFKA_DEBUG_ENABLE, mohaConf.getKafkaDebugEnable());
		env.put(MOHA_Properties.MYSQL_DEBUG_ENABLE, mohaConf.getMysqlLogEnable());
		env.put(MOHA_Properties.CONF_ZOOKEEPER_CONNECT, mohaConf.getZookeeperConnect());
		env.put(MOHA_Properties.CONF_ZOOKEEPER_BOOTSTRAP_SERVER, mohaConf.getBootstrapServers());

		String zookeeperConnect = mohaConf.getZookeeperConnect() + "/" + MOHA_Properties.ZOOKEEPER_ROOT_KAFKA + "/" + mohaConf.getKafkaClusterId();
		String bootstrapServer = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_KAFKA, mohaConf.getKafkaClusterId()).getBootstrapServers();

		env.put(MOHA_Properties.KAFKA_ZOOKEEPER_CONNECT, zookeeperConnect);
		env.put(MOHA_Properties.KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER, bootstrapServer);
		try {
			setEnv(env);
		} catch (Exception e3) {
			// TODO Auto-generated catch block
			e3.printStackTrace();
		}
		ApplicationSubmissionContext appContext = yarnClientApplication.getApplicationSubmissionContext();
		appContext.setApplicationName(appName);

		ContainerLaunchContext managerContainer = Records.newRecord(ContainerLaunchContext.class);
		LOG.info("Local resources = {}", localResources.toString());
		managerContainer.setLocalResources(localResources);
		managerContainer.setEnvironment(env);
		LOG.info("Environment variables = {}", env.toString());
		Vector<CharSequence> vargs = new Vector<>();
		vargs.add(Environment.JAVA_HOME.$() + "/bin/java");
		vargs.add(MOHA_Manager.class.getName());
		vargs.add(appId.toString());
		vargs.add(String.valueOf(executorMemory));
		vargs.add(String.valueOf(numExecutors));
		vargs.add(String.valueOf(startingTime));

		vargs.add("1><LOG_DIR>/MOHA_Manager.stdout");
		vargs.add("2><LOG_DIR>/MOHA_Manager.stderr");
		StringBuilder command = new StringBuilder();
		for (CharSequence str : vargs) {
			command.append(str).append(" ");
		}
		List<String> commands = new ArrayList<>();
		commands.add(command.toString());

		LOG.info("Command to execute MOHA Manager = {}", command);

		managerContainer.setCommands(commands);

		Resource capability = Records.newRecord(Resource.class);
		capability.setMemory(managerMemory);
		appContext.setResource(capability);
		appContext.setAMContainerSpec(managerContainer);

		Priority pri = Records.newRecord(Priority.class);
		pri.setPriority(priority);
		appContext.setPriority(pri);
		appContext.setQueue(queue);

		MOHA_Zookeeper zks = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_MOHA, appId.toString());

		zks.createRoot();
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_EXECUTORS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_LOGS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_STATUS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_TIME);
		for (int i = 0; i < numExecutors; i++) {
			zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_TIME + "/" + i);
		}

		/* Creating a queue and pushing jobs to the queue */

		MOHA_Queue jobQueue = new MOHA_Queue(zookeeperConnect, bootstrapServer, appId.toString());
		LOG.info(jobQueue.toString());

		jobQueue.create(numExecutors, 1);		
		
		jobQueue.register();

		/* Push task commands to the queue */
		FileReader fileReader;
		int numCommands;
		String jobCommands;
		try {
			fileReader = new FileReader(this.jdlPath);
			LOG.info(fileReader.toString());
			BufferedReader buff = new BufferedReader(fileReader);
			numCommands = Integer.parseInt(buff.readLine());
			jobCommands = buff.readLine();
			buff.close();

			for (int i = 0; i < numCommands; i++) {
				// jobQueue.push(Integer.toString(i), jobCommands);
				LOG.info(jobCommands);
			}

		} catch (NumberFormatException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		List<PartitionInfo> infor = jobQueue.partitionsFor();

		LOG.info(String.valueOf(infor.size()));
		for (int i = 0; i < infor.size(); i++) {
			LOG.info(infor.get(i).toString());
		}

		try {
			Thread.sleep(200);
		} catch (InterruptedException e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}

		File folder = new File("vina_input");
		File[] listOfFiles = folder.listFiles();
		Path tarsrc;
		String tarPathSuffix;
		Path tarDest;
		Path[] uris = new Path[listOfFiles.length];

		for (int i = 0; i < listOfFiles.length; i++) {
			if (listOfFiles[i].isFile()) {
				System.out.println("File " + listOfFiles[i].getName());
				tarsrc = new Path("vina_input/" + listOfFiles[i].getName());
				tarPathSuffix = rootDir + "/" + "vina_input/" + listOfFiles[i].getName();
				tarDest = new Path(fs.getHomeDirectory(), tarPathSuffix);
				fs.copyFromLocalFile(false, true, tarsrc, tarDest);
				jobQueue.push(Integer.toString(i), tarDest.toString());
				uris[i] = tarDest;
			} else if (listOfFiles[i].isDirectory()) {
				System.out.println("Directory " + listOfFiles[i].getName());
			}
		}

		/* Submits the application */

		LOG.info("MOHA Manager Container = {}", managerContainer.toString());
		ApplicationId appId = yarnClient.submitApplication(appContext);
		LOG.info("AppID = {}", appId.toString());
		LOG.info(zookeeperConnect + bootstrapServer);
		MOHA_Logger logger = new MOHA_Logger(MOHA_Client.class, Boolean.parseBoolean(mohaConf.getKafkaDebugEnable()), mohaConf.getDebugQueueName(),
				zookeeperConnect, bootstrapServer , appId.toString());
		logger.init();
		logger.subcribe();
		
		zks.setStatus(true);

		// Waiting for MOHA_Manager fully launched
		try {
			Thread.sleep(MOHA_Properties.MOHA_MANAGER_OVERHEAD);
		} catch (InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		// Checking MOHA_Manager status if it is running
		while (zks.getStatus()) {
			try {
				zks.setStatus(false);
				List<String> logs= logger.getLogs();
				for(String log:logs){
					LOG.info(log);
				}
				Thread.sleep(3000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}
		jobQueue.deleteQueue();
		zks.close();
		for (int i = 0; i < uris.length; i++) {
			fs.delete(uris[i], true);
		}
		LOG.info("Deleting root directory which contains jar file and jdl file on hdfs : {}", rootDir);
		fs.delete(new Path(rootDir), true);
		fs.close();
		logger.close();
		LOG.info("Application successfully finish");
		return true;
	}// The end of run function
	
	public static void setEnv(Map<String, String> newenv) throws Exception {
	    Class[] classes = Collections.class.getDeclaredClasses();
	    Map<String, String> env = System.getenv();
	    for(Class cl : classes) {
	        if("java.util.Collections$UnmodifiableMap".equals(cl.getName())) {
	            Field field = cl.getDeclaredField("m");
	            field.setAccessible(true);
	            Object obj = field.get(env);
	            Map<String, String> map = (Map<String, String>) obj;
	            map.clear();
	            map.putAll(newenv);
	        }
	    }
	}

}// The end of MOHA_Client class

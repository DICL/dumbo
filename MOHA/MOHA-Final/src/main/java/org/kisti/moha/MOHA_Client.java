package org.kisti.moha;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;

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
	private String configPath;
	private long startingTime;

	String rootDir;
	String zookeeperConnect;
	String bootstrapServer;
	MOHA_Zookeeper zks;
	MOHA_Configuration mohaConf;

	FileReader fileReader;
	String appDependencyFiles = "";
	int numCommands = 0;
	String jobCommands = "";
	String appType = "";
	MOHA_Queue jobQueue;
	boolean pushingFinish = false;
	boolean rStop = false;

	public static void main(String[] args) throws IOException {
		// [UPDATE] Change the flow of the main function to enable exception
		// handling at each step
		/*
		 * MOHA_Client client = new MOHA_Client(args); try { boolean result = client.run(); LOG.info(String.valueOf(result)); } catch (YarnException e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); } catch (IOException e) { // TODO Auto-generated catch block e.printStackTrace(); }
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
		 * try { LOG.info("Start init MOHA_Client"); startingTime = System.currentTimeMillis(); init(args); LOG.info("Successfully init"); conf = new YarnConfiguration(); yarnClient =
		 * YarnClient.createYarnClient(); yarnClient.init(conf); fs = FileSystem.get(conf); } catch (ParseException e) { // TODO Auto-generated catch block e.printStackTrace(); }
		 */
		conf = new YarnConfiguration();
		yarnClient = YarnClient.createYarnClient();
		yarnClient.init(conf);
		fs = FileSystem.get(conf);
	}// The end of MOHA_Client constructor

	public boolean init(String[] args) throws ParseException {
		/*
		 * Add an option that only contains a short-name. It may be specified as requiring an argument. - Parameters : opt (Short single-character name of the option) : hasArg flag (signally if an
		 * argument is required after this option) : description (Self-documenting description) - Returns: the resulting Options instance
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
		option.addOption("conf", true, "Job Description Language file that contains the MOHA job specification (must specified)");
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
		configPath = inputParser.getOptionValue("conf", "conf/MOHA.conf");

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

		LOG.info("App name = {}, priority = {}, queue = {}, manager memory = {}, jarPath = {}, executor memory = {}, " + "num ececutors = {}, jdl path = {}, conf path = {}", appName, priority, queue,
				managerMemory, jarPath, executorMemory, numExecutors, jdlPath, configPath);

		return true;
	}// The end of init function

	private void printUsage(Options opts) {
		new HelpFormatter().printHelp("MOHA_Client", opts);
	}

	/*
	 * private void dispatchDockingTasks(MOHA_Queue jobQueue, String rootDir) { File folder = new File("vina_input"); File[] listOfFiles = folder.listFiles(); Path tarsrc; String tarPathSuffix; Path
	 * tarDest; Path[] uris = new Path[listOfFiles.length]; for (int i = 0; i < listOfFiles.length; i++) { if (listOfFiles[i].isFile()) { System.out.println("File " + listOfFiles[i].getName()); tarsrc
	 * = new Path("vina_input/" + listOfFiles[i].getName()); tarPathSuffix = rootDir + "/" + "vina_input/" + listOfFiles[i].getName(); tarDest = new Path(fs.getHomeDirectory(), tarPathSuffix);
	 * System.out.println("tarDest:" + tarDest); try { fs.copyFromLocalFile(false, true, tarsrc, tarDest); } catch (IOException e) { // TODO Auto-generated catch block e.printStackTrace(); }
	 * jobQueue.push(Integer.toString(i), tarDest.toString()); uris[i] = tarDest; } else if (listOfFiles[i].isDirectory()) { System.out.println("Directory " + listOfFiles[i].getName()); } } }
	 */
	private void updateToDataset(String ins) throws Exception {
		DataSource source = new DataSource("cpu.arff");
		Instances inst = addToDataset(source.getDataSet(), ins);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(inst);
		saver.setFile(new File("cpu.arff"));
		saver.writeBatch();

	}

	private String getResourceStatus() {
		List<String> command = new ArrayList<String>();
		String data = "";
		command.add("cat");
		command.add("/proc/cpuinfo");
		// command.add(kbInfo.getKafkaBinDir() + "/config/" + MOHA_Properties.KAFKA_SERVER_PROPERTIES);

		ProcessBuilder builder = new ProcessBuilder(command);
		Process p;
		String line;
		LOG.debug("ProcessBuilder builder = " + command);
		try {
			p = builder.start();
			System.out.println("p = builder.start();");
			// try {
			// p.waitFor();
			// } catch (InterruptedException e) {
			// // TODO Auto-generated catch block
			// e.printStackTrace();
			// }

			// Thread.sleep(2000);
			// LOG.debug("p.waitFor()");
			BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String cpuMhz = "", numCpu = "";
			while ((line = br.readLine()) != null) {
				// System.out.println(line);
				if (line.contains("cpu MHz")) {
					cpuMhz = line.substring(line.lastIndexOf(":") + 1);
					// System.out.println(line.substring(line.lastIndexOf(":") + 1));
				}
				if (line.contains("processor")) {
					numCpu = line.substring(line.lastIndexOf(":") + 1);
					// System.out.println(line.substring(line.lastIndexOf(":") + 1));
				}
			}
			System.out.println(numCpu + "  " + cpuMhz);

			command = new ArrayList<String>();
			command.add("cat");
			command.add("/proc/meminfo");

			builder = new ProcessBuilder(command);
			LOG.debug("ProcessBuilder builder = " + command);

			p = builder.start();
			System.out.println("p = builder.start();");
			// try {
			// p.waitFor();
			// } catch (InterruptedException e) {
			// // TODO Auto-generated catch block
			// e.printStackTrace();
			// }

			// Thread.sleep(2000);
			// LOG.debug("p.waitFor()");
			br = new BufferedReader(new InputStreamReader(p.getInputStream()));
			String MemTotal = "", MemFree = "", Buffers = "", Cached = "", Active = "", Inactive = "";
			while ((line = br.readLine()) != null) {
				// System.out.println(line);
				if (line.contains("MemTotal:")) {
					MemTotal = line.substring(line.lastIndexOf(":") + 1).replace("kB", "").replace(" ","");
					//System.out.println(MemTotal);
				}
				if (line.contains("MemFree:")) {
					MemFree = line.substring(line.lastIndexOf(":") + 1).replace("kB", "").replace(" ","");
					//System.out.println(MemFree);
				}
				if (line.contains("Buffers:")) {
					Buffers = line.substring(line.lastIndexOf(":") + 1).replace("kB", "").replace(" ","");
					//System.out.println(Buffers);
				}
				if (line.contains("Cached:")&&!line.contains("wap")) {
					Cached = line.substring(line.lastIndexOf(":") + 1).replace("kB", "").replace(" ","");
					//System.out.println(Cached);
				}
				if (line.contains("Active:")) {
					Active = line.substring(line.lastIndexOf(":") + 1).replace("kB", "").replace(" ","");
					//System.out.println(Active);
				}
				if (line.contains("Inactive:")) {
					Inactive = line.substring(line.lastIndexOf(":") + 1).replace("kB", "").replace(" ","");
					//System.out.println(Inactive);
				}

			}
			
			
			command = new ArrayList<String>();
			command.add("top");
			command.add("-bn1");

			builder = new ProcessBuilder(command);
			//LOG.debug("ProcessBuilder builder = " + command);

			p = builder.start();
			System.out.println("p = builder.start();");
			// try {
			// p.waitFor();
			// } catch (InterruptedException e) {
			// // TODO Auto-generated catch block
			// e.printStackTrace();
			// }

			// Thread.sleep(2000);
			// LOG.debug("p.waitFor()");
			br = new BufferedReader(new InputStreamReader(p.getInputStream()));
			//String MemTotal = "", MemFree = "", Buffers = "", Cached = "", Active = "", Inactive = "";
			boolean collect = false;
			String [] arrOfStr;
			double cpuUsage = 0, memUsage = 0;
			String totalTasks = "0", runningTasks = "0";
			while ((line = br.readLine()) != null) {	 
				 line = line.trim();
				 //System.out.println(line);	
				 if((!collect)&&line.contains("Tasks:")){
					 line = line.replace(",", "");
					 line = line.replace(":", "");
					 arrOfStr = line.split("\\s+");
					 if(arrOfStr.length > 5){
						 totalTasks = arrOfStr[1];
						 runningTasks = arrOfStr[3];
					 }
				 }
				 if(collect){
					 arrOfStr = line.split("\\s+");
					 if(arrOfStr.length > 10){
						 //System.out.println(arrOfStr[8]);
						 //System.out.println(arrOfStr[9]);
						 cpuUsage += Double.valueOf(arrOfStr[8]);
						 memUsage += Double.valueOf(arrOfStr[9]);
					 }
					 
				 }			
				 
				 if(line.contains("PID"))collect = true;
				 
			}
			//System.out.println("CPU usage: " + cpuUsage +  "  memory usage: " + memUsage);
			//System.out.println("totalTasks: " + totalTasks +  "  runningTasks: " + runningTasks);
			
			data = numCpu + " " + cpuMhz + " " + MemTotal  + " "+ MemFree  + " "+ Buffers  + " "+ Active 
					+ " "+ Inactive  + " "+ cpuUsage  + " "+ memUsage  + " "+ totalTasks  + " "+ runningTasks  + " ";
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println(data);
		return data;
	}

	private void removeInstanceFromDataset() throws Exception {
		DataSource source = new DataSource("cpu.arff");
		Instances inst = source.getDataSet();
		int min = 0;
		int max = inst.numInstances() - 1;
		Random r = new Random();

		while (inst.numInstances() > 5) {

			int index = r.nextInt((max - min) + 1) + min;
			inst.delete(index);
			max = inst.numInstances() - 1;
		}

		ArffSaver saver = new ArffSaver();
		saver.setInstances(inst);
		saver.setFile(new File("cpu.arff"));
		saver.writeBatch();
	}

	private Instances addToDataset(Instances inst, String ins) {
		// Instance data = inst.lastInstance();

		String[] in = ins.split(" ");

		double[] instanceValue = new double[inst.numAttributes()];
		// DenseInstance denseInstance = new DenseInstance(1.0, in);
		for (int i = 0; i < in.length; i++) {
			instanceValue[i] = Double.parseDouble(in[i]);
		}

		DenseInstance denseInstance = new DenseInstance(1.0, instanceValue);

		inst.add(denseInstance);
		return inst;
	}

	private void wekaClassifier() throws Exception {
		DataSource source = new DataSource("diabetes_training.arff");
		// DataSource source = new DataSource("D:/eclipse/workspace/cpu.arff");
		Instances trainDataset = source.getDataSet();
		System.out.println("This is the training data \n");
		System.out.println(trainDataset.toString());
		trainDataset.setClassIndex(trainDataset.numAttributes() - 1);

		int numClasses = trainDataset.numClasses();
		for (int i = 0; i < numClasses; i++) {

			String classValue = trainDataset.classAttribute().value(i);
			System.out.println("Class Value" + i + " is" + classValue);
		}

		NaiveBayes nb = new NaiveBayes();

		nb.buildClassifier(trainDataset);

		// load new dataset
		source = new DataSource("diabetes_test.arff");
		Instances testDataset = source.getDataSet();
		testDataset.setClassIndex(testDataset.numAttributes() - 1);

		System.out.println("Actual Class, NB Predicted");

		for (int i = 0; i < testDataset.numInstances(); i++) {

			double actualClass = testDataset.instance(i).classValue();
			String actual = testDataset.classAttribute().value((int) actualClass);

			Instance newInst = testDataset.instance(i);

			double predNB = nb.classifyInstance(newInst);

			String preString = testDataset.classAttribute().value((int) predNB);

			System.out.println(actual + ", " + preString);

		}

		System.out.println(nb.getCapabilities().toString());
		if (true)
			return;
	}

	public boolean run() throws YarnException, IOException {
/*
		try {
			// wekaClassifier();
			for (int i = 0; i < 100; i++) {

				updateToDataset(String.valueOf(i) + " 1 2 3 4 5");
			}

		} catch (Exception e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}

		try {
			removeInstanceFromDataset();
		} catch (Exception e2) {
			// TODO Auto-generated catch block
			e2.printStackTrace();
		}

		getResourceStatus();

		if (true)
			return true;
*/
		LOG.info("yarnClient = {}", yarnClient.toString());
		// Get configuration information
		// mohaConf = new MOHA_Configuration("conf/MOHA.conf");
		mohaConf = new MOHA_Configuration(configPath);
		LOG.info("Configuration file = {}", mohaConf.toString());

		yarnClient.start();
		YarnClientApplication yarnClientApplication = yarnClient.createApplication();
		GetNewApplicationResponse appResponse = yarnClientApplication.getNewApplicationResponse();
		appId = appResponse.getApplicationId();

		rootDir = appName + "/" + appId;
		String hdfsLocalResoure = "";

		if (mohaConf.getQueueType().equals("kafka")) {
			zookeeperConnect = mohaConf.getZookeeperConnect() + "/" + MOHA_Properties.ZOOKEEPER_ROOT + "/" + mohaConf.getKafkaClusterId();
			bootstrapServer = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT, mohaConf.getKafkaClusterId()).getBootstrapServers();
		}

		/* Insert tasks to job queue */
		TaskSubmitter taskSubmitter = new TaskSubmitter();
		Thread taskSubmitterThread = new Thread(taskSubmitter);
		taskSubmitterThread.start();

		zks = new MOHA_Zookeeper(MOHA_Client.class, MOHA_Properties.ZOOKEEPER_ROOT, appId.toString());

		zks.createZooMOHA();// Create "moha" directory in zookeeper
		zks.createRoot();
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_EXECUTORS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_LOGS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_STATUS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_APP_LOG);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_TIME_START);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_TIMER);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_TIME_COMPLETE);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_NUM_PROCESSED_TASKS);
		zks.createDirs(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_STOP);

		/* Update current time to zookeeper server */
		Timer timer = new Timer();
		Thread timerThread = new Thread(timer);
		timerThread.start();

		// dispatchDockingTasks(jobQueue,rootDir);

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

		Map<String, String> env = new HashMap<>();

		Path jarSrc = new Path(this.jarPath);

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

		env.put(MOHA_Properties.APP_JAR, jarDest.toUri().toString());
		env.put(MOHA_Properties.APP_JAR_TIMESTAMP, Long.toString(jarDestStatus.getModificationTime()));
		env.put(MOHA_Properties.APP_JAR_SIZE, Long.toString(jarDestStatus.getLen()));

		if (appType.equals("S")) {
			Path mohaAppSrc = new Path(appDependencyFiles);
			String pathSuffixmoha_tgz = appName + "/" + appId.getId() + "/" + appDependencyFiles;
			Path appDependencyDest = new Path(fs.getHomeDirectory(), pathSuffixmoha_tgz);
			fs.copyFromLocalFile(false, true, mohaAppSrc, appDependencyDest);
			FileStatus userAppStatus = fs.getFileLinkStatus(appDependencyDest);
			LOG.info("FileStatus ={}", userAppStatus.toString());
			hdfsLocalResoure = appDependencyDest.toUri().toString();

			env.put(MOHA_Properties.MOHA_TGZ, appDependencyDest.toUri().toString());
			env.put(MOHA_Properties.MOHA_TGZ_TIMESTAMP, Long.toString(userAppStatus.getModificationTime()));
			env.put(MOHA_Properties.MOHA_TGZ_SIZE, Long.toString(userAppStatus.getLen()));
		}

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
		env.put(MOHA_Properties.CONF_KAKFA_VERSION, mohaConf.getKafkaVersion());
		env.put(MOHA_Properties.CONF_KAFKA_CLUSTER_ID, mohaConf.getKafkaClusterId());
		env.put(MOHA_Properties.CONF_KAFKA_DEBUG_QUEUE_NAME, mohaConf.getDebugQueueName());
		env.put(MOHA_Properties.CONF_KAFKA_DEBUG_ENABLE, mohaConf.getKafkaDebugEnable());
		env.put(MOHA_Properties.CONF_MYSQL_DEBUG_ENABLE, mohaConf.getMysqlLogEnable());
		env.put(MOHA_Properties.CONF_ZOOKEEPER_CONNECT, mohaConf.getZookeeperConnect());
		env.put(MOHA_Properties.CONF_ZOOKEEPER_BOOTSTRAP_SERVER, mohaConf.getBootstrapServers());
		env.put(MOHA_Properties.CONF_ZOOKEEPER_SERVER, mohaConf.getZookeeperServer());
		env.put(MOHA_Properties.CONF_QUEUE_TYPE, mohaConf.getQueueType());
		env.put(MOHA_Properties.CONF_ACTIVEMQ_SERVER, mohaConf.getActivemqServer());

		env.put(MOHA_Properties.HDFS_HOME_DIRECTORY, fs.getHomeDirectory().toString());
		env.put(MOHA_Properties.APP_TYPE, appType);

		if (mohaConf.getQueueType().equals("kafka")) {
			env.put(MOHA_Properties.KAFKA_ZOOKEEPER_CONNECT, zookeeperConnect);
			env.put(MOHA_Properties.KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER, bootstrapServer);
		}
		try {
			setEnv(env);
		} catch (Exception e3) {
			// TODO Auto-generated catch block
			e3.printStackTrace();
		}
		ApplicationSubmissionContext appContext = yarnClientApplication.getApplicationSubmissionContext();
		appContext.setApplicationName(appName);

		ContainerLaunchContext managerContainer = Records.newRecord(ContainerLaunchContext.class);
		// LOG.info("Local resources = {}", localResources.toString());
		managerContainer.setLocalResources(localResources);
		managerContainer.setEnvironment(env);
		// LOG.info("Environment variables = {}", env.toString());
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

		// LOG.info("Command to execute MOHA Manager = {}", command);

		managerContainer.setCommands(commands);

		Resource capability = Records.newRecord(Resource.class);
		capability.setMemory(managerMemory);
		appContext.setResource(capability);
		appContext.setAMContainerSpec(managerContainer);

		Priority pri = Records.newRecord(Priority.class);
		pri.setPriority(priority);
		appContext.setPriority(pri);
		appContext.setQueue(queue);

		while (!pushingFinish) {
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		/* Submits the application */

		// LOG.info("MOHA Manager Container = {}", managerContainer.toString());
		LOG.info("Submit the app [at]" + zks.getCurrentTime());
		ApplicationId appId = yarnClient.submitApplication(appContext);
		LOG.info("Submit Application - AppID = {}", appId.toString());
		// LOG.info("zookeeperConnect = {} bootstrapServer = {}", zookeeperConnect, bootstrapServer);

		zks.setLocalResource(hdfsLocalResoure);
		zks.setManagerRunning(true);
		zks.setStopRequest(false);

		// Waiting for MOHA_Manager fully launched
		try {
			Thread.sleep(MOHA_Properties.MOHA_MANAGER_OVERHEAD);
		} catch (InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		for (int i = 0; i < 100; i++) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			String logs = zks.getLogs();
			if (logs.length() > 0) {
				System.out.println(logs);
			}

		}

		int numOfProcessedTasks = 0;
		// Checking MOHA_Manager status if it is running
		while (zks.isManagerRunning()) {
			try {
				zks.setManagerRunning(false);
				for (int i = 0; i < 20; i++) {
					Thread.sleep(100);
					String _logs = zks.getLogs();
					if (_logs.length() > 0) {
						System.out.println(_logs);
					}
					numOfProcessedTasks = zks.getNumOfProcessedTasks();
					if ((numOfProcessedTasks > 0) && (numOfProcessedTasks < numCommands)) {
						System.out.println("Number of processed tasks is " + numOfProcessedTasks + "/" + numCommands);
					}
					// Get timeout from Task Executors
					if (zks.getStopRequest()) {
						if (numOfProcessedTasks < numCommands) {
							System.out.println("MOHA Client: Timeout");
						}
						break;
					}
					// All tasks are processed
					if (numOfProcessedTasks >= numCommands) {
						zks.setStopRequest(true);
						System.out.println("Number of processed tasks is " + numOfProcessedTasks + "/" + numCommands);
						System.out.println("MOHA Client: All tasks are processed");
						break;
					}
				}

			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}

		}

		try {
			Thread.sleep(5000);
		} catch (InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		String logs;

		logs = zks.getResultsExe();
		if (logs.length() > 0) {
			System.out.println(logs);
		}

		try {
			Files.write(Paths.get("results/exe.log"), "\n".getBytes(), StandardOpenOption.APPEND);
			Files.write(Paths.get("results/exe.log"), logs.getBytes(), StandardOpenOption.APPEND);
		} catch (IOException e) {
			// exception handling left as an exercise for the reader
		}

		logs = zks.getPerformanceExe();
		if (logs.length() > 0) {
			// System.out.println(logs);
		}

		try {
			Files.write(Paths.get("results/exe_" + appId + mohaConf.getQueueType() + "_appType_" + appType + ".log"), logs.getBytes(), StandardOpenOption.CREATE_NEW);
		} catch (IOException e) {
			// exception handling left as an exercise for the reader
			System.out.println(e.toString());
		}

		// Read results from MOHA manager and task executors
		double executionTime = (double) ((Long.valueOf(zks.getTimeComplete()) - Long.valueOf(zks.getTimeStart())));
		double performance = (double) numOfProcessedTasks * 1000 / executionTime;
		logs = MOHA_Common.convertLongToDate(System.currentTimeMillis()) + " " + appId.toString() + " " + String.valueOf(numExecutors) + " " + String.valueOf(executorMemory) + " numOfProcessedTasks: "
				+ String.valueOf(numOfProcessedTasks) + " Queue: " + mohaConf.getQueueType() + " appType: " + appType + " Time Start: " + zks.getTimeStart() + " Time Finish: " + zks.getTimeComplete()
				+ " ExecutionTime: " + executionTime + " Performance: " + performance + "\n";

		try {
			Files.write(Paths.get("results/app.log"), logs.getBytes(), StandardOpenOption.APPEND);
			System.out.println(logs);
		} catch (IOException e) {
			// exception handling left as an exercise for the reader
		}
		System.out.println("End loop");
		rStop = true;
		jobQueue.deleteQueue();
		jobQueue.close();
		zks.delete(zks.getRoot());
		zks.close();
		LOG.info("Deleting root directory which contains jar file and jdl file on hdfs : {}", rootDir);
		fs.delete(new Path(rootDir), true);
		fs.close();
		LOG.info("Application successfully finish");
		return true;
	}// The end of run function
	/*
	 * private boolean copyFolderToHDFS(String directory, String destRoot) { File dir = new File(directory); File[] fileList = dir.listFiles(); String hdfsDest; for (int i = 0; i < fileList.length;
	 * i++) { if (fileList[i].isFile()) { System.out.println("File " + fileList[i].getName()); hdfsDest = destRoot + "/" + fileList[i].getName(); try { fs.copyFromLocalFile(false, true, new
	 * Path(directory, fileList[i].getName()), new Path(fs.getHomeDirectory(), hdfsDest)); } catch (IOException e) { // TODO Auto-generated catch block e.printStackTrace(); } } else if
	 * (fileList[i].isDirectory()) { System.out.println("Directory " + fileList[i].getName()); } } return true; }
	 */

	public static void setEnv(Map<String, String> newenv) throws Exception {
		Class[] classes = Collections.class.getDeclaredClasses();
		Map<String, String> env = System.getenv();
		for (Class cl : classes) {
			if ("java.util.Collections$UnmodifiableMap".equals(cl.getName())) {
				Field field = cl.getDeclaredField("m");
				field.setAccessible(true);
				Object obj = field.get(env);
				Map<String, String> map = (Map<String, String>) obj;
				map.clear();
				map.putAll(newenv);
			}
		}
	}

	protected class Timer implements Runnable {
		public Timer() {

		}

		@Override
		public void run() {
			// TODO Auto-generated method stub
			while (!rStop) {
				zks.setCurrentTime(System.currentTimeMillis());
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
			}

		}

	}

	protected class TaskSubmitter implements Runnable {

		public TaskSubmitter() {
			// TODO Auto-generated constructor stub
		}

		@Override
		public void run() {
			// TODO Auto-generated method stub
			System.out.println("Start pushing tasks to the job queue");
			if (mohaConf.getQueueType().equals("kafka")) {
				jobQueue = new MOHA_Queue(zookeeperConnect, bootstrapServer, appId.toString());
			} else {
				jobQueue = new MOHA_Queue(mohaConf.getActivemqServer(), appId.toString());
			}

			jobQueue.queueCreate(numExecutors, 1);

			jobQueue.producerInit();

			/* Push task commands to the queue */
			long starting_time = System.currentTimeMillis();

			try {
				// The file that contains description of user application
				fileReader = new FileReader(jdlPath);
				LOG.info(fileReader.toString());
				BufferedReader buff = new BufferedReader(fileReader);
				appType = buff.readLine();

				if (appType.equals("S")) {// S denotes parameter sweeping application
					System.out.println("S");
					int num_inputs = Integer.parseInt(buff.readLine());
					List<String> directories = new ArrayList<String>();
					List<String> order = new ArrayList<String>();
					List<String> parameters = new ArrayList<String>();
					String shell_command;

					for (int i = 0; i < num_inputs; i++) {
						String temp = buff.readLine();
						String[] str = temp.split(" ");
						System.out.println("str[0]: " + str[0]);
						System.out.println("str[1]: " + str[1]);
						order.add(str[0]);
						switch (str[0]) {
						case "D":// D denotes directory which contains many input files
							directories.add(str[1]);
							break;
						case "P":// P denotes file which contains input parameters
							parameters.add(str[1]);
							break;
						default:
							System.out.println("switch case default:" + str[0]);
							break;
						}
					}
					shell_command = buff.readLine(); // the executable shell script file name
					appDependencyFiles = buff.readLine();// the compressed file that contains all related files, library etc
					String command_description = null;
					// Copy all input files in the folder to HDFS
					LOG.info("Start copying local files to HDFS");
					for (int i = 0; i < directories.size(); i++) {
						// copyFolderToHDFS(directory.get(i), rootDir);
						Path local_source = new Path(directories.get(i));
						Path hdfs_dest = new Path(fs.getHomeDirectory(), rootDir + "/" + directories.get(i));
						fs.copyFromLocalFile(false, true, local_source, hdfs_dest);
						LOG.info("Copying local files to HDFS " + directories.get(i));
					}
					pushingFinish = true;

					double time = (double) (System.currentTimeMillis() - starting_time) / 1000;
					System.out.println("Copying to HDFS time: " + time + " seconds");

					LOG.info("Start sending commands to Task Executors");

					/* Push task commands to the queue */
					starting_time = System.currentTimeMillis();

					if (directories.size() == 2) {
						List<String[]> file_array = new ArrayList<String[]>();

						File[] fileList0 = new File(directories.get(0)).listFiles();
						File[] fileList1 = new File(directories.get(1)).listFiles();
						int q_index = 0;
						for (int i = 0; i < fileList0.length; i++) {
							if (fileList0[i].isFile()) {
								for (int j = 0; j < fileList1.length; j++) {
									if (fileList1[j].isFile()) {
										String[] fileName = new String[2];
										fileName[0] = fileList0[i].getPath();
										fileName[1] = fileList1[j].getPath();
										file_array.add(fileName);
										command_description = "";
										command_description += String.valueOf(num_inputs) + " ";
										int f_index = 0;
										int p_index = 0;

										for (int k = 0; k < num_inputs; k++) {
											switch (order.get(k)) {
											case "D":
												command_description += "D " + fileName[f_index] + " ";
												f_index += 1;
												break;
											case "P":
												command_description += "P " + parameters.get(p_index) + " ";
												p_index += 1;
												break;
											default:
												// do nothing
												break;
											}
										}
										command_description += shell_command;
										// System.out.println("Command send to Task Executor: " + command_description);
										jobQueue.push(Integer.toString(q_index), command_description);
										q_index++;
										numCommands++;
									}
								}
							} else if (fileList0[i].isDirectory()) {
								System.out.println("Directory " + fileList0[i].getName());
							}
						}

					}
					String[] command_compo = command_description.split(" ");
					for (int i = 0; i < command_compo.length - 1; i++) {
						System.out.println(command_compo[i]);
					}

					String run_command = "./";
					run_command += command_compo[2 * Integer.valueOf(command_compo[0]) + 1];

					for (int i = 1; i <= Integer.valueOf(command_compo[0]); i++) {
						/*
						 * if(real_command[i*2-1].equals("D")){ System.out.println(real_command[i*2]); //String[] file_name_only = real_command[i*2].split("\\.");
						 * //System.out.println(file_name_only.toString()); //run_command += " " + file_name_only[0]; run_command += " " + real_command[i*2]; }else{ run_command += " " +
						 * real_command[i*2]; }
						 */
						run_command += " " + command_compo[i * 2];

					}
					System.out.println("Run command: " + run_command);

					if (parameters.size() < 0) {
						buff.close();
						fileReader.close();
						return;
					}

				} else {// this is for executing same task multiple times
					// Command mode
					System.out.println("App type: " + appType);
					numCommands = numExecutors * Integer.parseInt(buff.readLine());
					jobCommands = buff.readLine();
					System.out.println("Start pushing tasks to the job queue, number of tasks: " + numCommands + " command: " + jobCommands);
					for (int i = 0; i < numCommands; i++) {
						jobQueue.push(Integer.toString(i), jobCommands);
						// LOG.info(jobCommands);
					}

				}
				buff.close();
				fileReader.close();
			} catch (NumberFormatException | IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			double pushing_performance = (double) numCommands * 1000 / (double) (System.currentTimeMillis() - starting_time);
			double seconds = (double) (System.currentTimeMillis() - starting_time) / 1000;
			System.out.println("Pushing performance: " + pushing_performance + " (tasks/second) on " + seconds + " seconds");

			return;
		}

	}

}// The end of MOHA_Client class

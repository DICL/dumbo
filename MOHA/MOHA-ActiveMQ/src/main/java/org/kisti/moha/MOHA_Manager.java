package org.kisti.moha;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerLaunchContext;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.api.records.FinalApplicationStatus;
import org.apache.hadoop.yarn.api.records.LocalResource;
import org.apache.hadoop.yarn.api.records.LocalResourceType;
import org.apache.hadoop.yarn.api.records.LocalResourceVisibility;
import org.apache.hadoop.yarn.api.records.NodeReport;
import org.apache.hadoop.yarn.api.records.Priority;
import org.apache.hadoop.yarn.api.records.Resource;
import org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync;
import org.apache.hadoop.yarn.client.api.async.AMRMClientAsync.CallbackHandler;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.hadoop.yarn.exceptions.YarnException;
import org.apache.hadoop.yarn.util.ConverterUtils;
import org.apache.hadoop.yarn.util.Records;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MOHA_Manager {	
	private static final Logger LOG = LoggerFactory.getLogger(MOHA_Manager.class);
	private YarnConfiguration conf;
	private AMRMClientAsync<ContainerRequest> amRMClient;
	private FileSystem fileSystem;
	//private int numOfContainers;

	private volatile boolean done = false;
	protected NMClientAsync nmClient;
	private NMCallbackHandler containerListener;
	private List<Thread> launchThreads = new ArrayList<>();
	
	private Options opts;
	private int mohaPriority = 0;
	private int executorMemory = 1024;
	private int numExecutors = 1;
	
	//For the commands in the JDL file
	private int numOfCommands = 0;
	private String shell_command = ""; 
	private String ApplicationID = "";
	
	//For the statistics
	private long manager_start_time;
	private long manager_init_complete_time;
	private long manager_container_request_finish_time;
	private long resource_allocation_finish_time;
	private long manager_task_completion_time;
	private long manager_finish_time;
	protected AtomicInteger numCompletedContainers = new AtomicInteger();
	private AtomicInteger numAllocatedContainers = new AtomicInteger();

	
	public MOHA_Manager (String [] args) throws IOException {
		//Record the start time of the MOHA_Manager
		manager_start_time = new Date().getTime();
		
		conf = new YarnConfiguration();
		fileSystem = FileSystem.get(conf);
		
		/*
		 * MOHA_Manager takes three arguments: priority, executor memory and # of executors
		 * The location of MOHA application jar file is passed through environment variables
		 */
		opts = new Options();
		
		opts.addOption("priority", true, "Application Priority (Default: 0)");
		opts.addOption("executor_memory", true, "Amount of memory in MB for MOHA TaskExecutor (Default: 1024)");
		opts.addOption("num_executors", true, "Number of MOHA Task Executors");
	}//The end of constructor
	
	
	public static void main(String[] args) {		
		boolean result;
				
		LOG.info("Starting the MOHA_Manager");
		
		if(args.length == 0) {
			LOG.error("Empty parameters given by MOHA_Client !");
			return;
		}
		
		try {
			MOHA_Manager manager = new MOHA_Manager(args);
			
			result = manager.init(args);
			if(!result) {
				LOG.error("Finishing the MOHA_Manager without executing containers");
				return;
			}
						
			manager.run();
			manager.printStatistics();
		} catch (IOException | YarnException | ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
				
		LOG.info("MOHA_Manager completes !");		
	}//The end of main function
	
	
	public boolean init(String[] args) throws ParseException {	
		/* Parse the arguments */
		CommandLine cliParser = new GnuParser().parse(opts, args);
		
		if(!cliParser.hasOption("priority") || !cliParser.hasOption("executor_memory") 
				|| !cliParser.hasOption("num_executors")) {
			LOG.error("Input parameters are illegally given by the MOHA_Client !");
			return false;			
		}
		
		mohaPriority = Integer.parseInt(cliParser.getOptionValue("priority", "0"));
		LOG.info("Priority of MOHA_Manager & TaskExecutor = {}", mohaPriority);
		
		executorMemory = Integer.parseInt(cliParser.getOptionValue("executor_memory", "1024"));
		LOG.info("Amount of memory in MB for MOHA TaskExecutor = {}", executorMemory);
				
		numExecutors = Integer.parseInt(cliParser.getOptionValue("num_executors", "1"));
		LOG.info("Number of MOHA Task Executors is set to {}", numExecutors);
	
		/* 
		 * Parse the JDL file
		 * - First line should contain the number of tasks to be executed
		 * - Second line should contain the shell command 
		 */		
		BufferedReader br = null;
		String line = "";
		String ActiveMQ_URL="";
		int lineNumber = 1;
		boolean result;
		
		try {
			br = new BufferedReader(new FileReader(MOHA_Constants.JDL_NAME));
			
			while ((line = br.readLine()) != null) {
				//The first line represents the number of shell commands
				if(lineNumber == 1) {
					this.numOfCommands = Integer.parseInt(line);
				}
				//The second line represents the shell command
				else if(lineNumber == 2) {
					this.shell_command = line.trim();
				}
				
				lineNumber++;
			}//The end of while
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		LOG.info("The shell commands to be executed on containers: {} of {}", this.numOfCommands, this.shell_command);
		
		//Check the location of ActiveMQ Service and the Application ID
		ActiveMQ_URL = System.getenv().get("ACTIVEMQ_URL");
		LOG.info("The location of ActiveMQ Service = {}", ActiveMQ_URL);
		
		ApplicationID = System.getenv().get("MOHA_APP_ID");
		LOG.info("The Application ID of MOHA_Manager = {}", this.ApplicationID);
		
		//Create a job queue and insert tasks
		result = LaunchQueue(this.numOfCommands, this.shell_command, ActiveMQ_URL);
		if(!result) {
			LOG.error("Launching Queue and Insertion of tasks have failed !");
			return false;
		}		
		
		manager_init_complete_time = new Date().getTime();
				
		return true;
	}//The end of init function
	
	
	public boolean LaunchQueue(int numOfCommands, String command, String queue_location) {
		boolean result;
		
		/* Insert tasks into the ActiveMQ */
		ActiveMQ_Manager amq_manager = new ActiveMQ_Manager(ApplicationID, queue_location, MOHA_Constants.ACTIVEMQ_PRODUCER);
		result = amq_manager.SimpleInsertTasks(numOfCommands, command);
		amq_manager.Finish_AMQ(MOHA_Constants.ACTIVEMQ_PRODUCER);
		
		return result;
	}//The end of InsertQueue function
	
	
	public void printStatistics() {
		double elapsedTime;
		
		elapsedTime = (double)(this.manager_finish_time - this.manager_start_time)/(double)1000;
		LOG.info("The total elapsed time (sec) is {}", elapsedTime);
		
		elapsedTime = (double)(this.manager_init_complete_time - this.manager_start_time)/(double)1000;
		LOG.info("The init time including launching queue (sec) is {}", elapsedTime);
		
		elapsedTime = (double)(this.manager_container_request_finish_time - this.manager_init_complete_time)/(double)1000;
		LOG.info("The time to request containers (sec) is {}", elapsedTime);
		
		elapsedTime = (double)(this.resource_allocation_finish_time - this.manager_container_request_finish_time)/(double)1000;
		LOG.info("The resource allocation time of {} executors (sec) is {}", this.numExecutors, elapsedTime);	
		
		elapsedTime = (double)(this.manager_finish_time - this.manager_task_completion_time)/(double)1000;
		LOG.info("The shutting down time of NMClient and AMRMClient (sec) is {}", elapsedTime);
	}//The end of printStatistics function
	
	
	public void run() throws YarnException, IOException {
		//setup the client for the ResourceManager
		amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, new RMCallbackHandler());
		amRMClient.init(conf);
		amRMClient.start();
		
		//register the application
		RegisterApplicationMasterResponse response;
		response = amRMClient.registerApplicationMaster(NetUtils.getHostname(), -1, "");
		//LOG.info("MOHA_Manager is registered with response: {}", response.toString());
				
		//setup the client for the NodeManager
		containerListener = new NMCallbackHandler(this);
		nmClient = NMClientAsync.createNMClientAsync(containerListener);
		nmClient.init(conf);
		nmClient.start();
		
		//request the containers
		Resource capacity = Records.newRecord(Resource.class);
		capacity.setMemory(executorMemory);
		Priority priority = Records.newRecord(Priority.class);
		priority.setPriority(mohaPriority);
		
		/* To specify prefered list of nodes where application containers will run */
		String[] prefered_nodes = new String[] {"hdp02.kisti.re.kr", "hdp03.kisti.re.kr"};
		
		for(int i = 1; i <= numExecutors; i++) {			
			/*
			 * org.apache.hadoop.yarn.client.api.AMRMClient.ContainerRequest.ContainerRequest
			 * (Resource capability, String[] nodes, String[] racks, Priority priority, boolean relaxLocality)
			 */
			//ContainerRequest ask = new ContainerRequest(capacity, prefered_nodes, null, priority, false);
			ContainerRequest ask = new ContainerRequest(capacity, null, null, priority);
			
			amRMClient.addContainerRequest(ask);
			//numOfContainers++;
		}
		
		//Record the time to complete container requests
		manager_container_request_finish_time = new Date().getTime();
		
		while(!done && numCompletedContainers.get() < numExecutors) {
			//LOG.info("The number of completed executors = " + this.numCompletedContainers.get());
			try {
				//Thread.sleep(2000);
				Thread.sleep(200);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		LOG.info("The number of completed executors = " + this.numCompletedContainers.get());
				
		/* Join all launched threads: needed for when we time
		   out and we need to release containers */
		for (Thread launchThread : launchThreads) {
			try {
				launchThread.join(10000);
			}
			catch (InterruptedException e) {
				LOG.info("Exception thrown in thread join: {}", e.getMessage());
				e.printStackTrace();
			}
		}
		
		manager_task_completion_time = new Date().getTime();
		LOG.info("Containers have all completed, so shutting down NMClient and AMRMClient...");
				
		nmClient.stop();		
		amRMClient.unregisterApplicationMaster(FinalApplicationStatus.SUCCEEDED, "MOHA_Manager complete!", null);		
		amRMClient.stop();
		
		//Record the finish time of the MOHA_Manager
		manager_finish_time = new Date().getTime();
	}//The end of run function
		
	
	protected class ContainerLauncher implements Runnable {
		private Container container;
		private NMCallbackHandler containerListener;
		
		public ContainerLauncher(Container container, NMCallbackHandler containerListener) {
			super();
			this.container = container;
			this.containerListener = containerListener;
		}

		@Override
		public void run() {
			LOG.info("Setting up ContainerLaunchContext for containerid = {}", container.getId());
			
			Map<String, LocalResource> localResources = new HashMap<>();
			Map<String, String> env = System.getenv();
			
			LocalResource appJarFile = Records.newRecord(LocalResource.class);
			appJarFile.setType(LocalResourceType.FILE);
			appJarFile.setVisibility(LocalResourceVisibility.APPLICATION);
			try {
				appJarFile.setResource(ConverterUtils.getYarnUrlFromURI(new URI(env.get("MOHA_JAR"))));
			} catch (URISyntaxException e) {
				e.printStackTrace();
				return;
			}
			appJarFile.setTimestamp(Long.valueOf((env.get("MOHA_JAR_TIMESTAMP"))));
			appJarFile.setSize(Long.valueOf(env.get("MOHA_JAR_LEN")));
			localResources.put(MOHA_Constants.APP_JAR_NAME, appJarFile);
			//LOG.info("Added {} as a local resource to the Container (Jar file)", appJarFile.toString());
			
			//LOG.info("CLASSPATH set by the MOHA_Manager: {}", env.get("CLASSPATH"));
								
			ContainerLaunchContext context = Records.newRecord(ContainerLaunchContext.class);
			context.setEnvironment(env);
			context.setLocalResources(localResources);
			String command = this.getLaunchCommand(container);
			List<String> commands = new ArrayList<>();
			commands.add(command);
			context.setCommands(commands);
			LOG.info("Command to execute MOHA_TaskExecutor = {}", command);
			
			//start the MOHA_TaskExecutor on the allocated container
			nmClient.startContainerAsync(container, context);
			LOG.info("MOHA_TaskExecutor {} is launched", container.getId());			
		}//The end of run function
		
		
		public String getLaunchCommand(Container container) {
			Vector<CharSequence> vargs = new Vector<>(30);
			//vargs.add(Environment.JAVA_HOME.$() + "/bin/java");
			vargs.add("java");
			vargs.add("org.kisti.moha.MOHA_TaskExecutor");
			vargs.add("1><LOG_DIR>/MOHA_TaskExecutor.stdout");
			vargs.add("2><LOG_DIR>/MOHA_TaskExecutor.stderr");
			StringBuilder command = new StringBuilder();
			for (CharSequence str : vargs) {
				command.append(str).append(" ");
			}
			return command.toString();			
		}//The end of getLaunchCommand function
		
	}//The end of ContainerLauncher class
	

	public class RMCallbackHandler implements AMRMClientAsync.CallbackHandler {

		@Override
		public void onContainersCompleted(List<ContainerStatus> statuses) {
			LOG.info("Response from RM for completed containers: count = {}", statuses.size());		
			for(ContainerStatus status : statuses) {
				numCompletedContainers.incrementAndGet();
				LOG.info("Container completed : {}", status.getContainerId());
			}
		}

		@Override
		public void onContainersAllocated(List<Container> containers) {
			LOG.info("Response from RM for allocated containers: count = {}", containers.size());
			for(Container container : containers) {
				LOG.info("Starting Container on {}", container.getNodeHttpAddress());
								
				ContainerLauncher launcher = new ContainerLauncher(container, containerListener);
				Thread thread = new Thread(launcher);
				thread.start();
				launchThreads.add(thread);		
				
				//To measure the time to complete all resource allocations
				numAllocatedContainers.incrementAndGet();
				if(numAllocatedContainers.get() == numExecutors) {
					resource_allocation_finish_time = new Date().getTime();					
				}
			}//The end of for
		}//The end of onContainersAllocated function

		@Override
		public void onShutdownRequest() {
			done = true;
		}

		@Override
		public void onNodesUpdated(List<NodeReport> updatedNodes) {}

		@Override
		public float getProgress() {
			//float progress = numOfContainers <= 0 ? 0 : (float) numCompletedContainers.get() / numOfContainers;
			float progress = numExecutors <= 0 ? 0 : (float) numCompletedContainers.get() / numExecutors;
			return progress;
		}

		@Override
		public void onError(Throwable e) {
			done = true;
			amRMClient.stop();
		}

	}//The end of RMCallbackHandler class
	
	
	public class NMCallbackHandler implements NMClientAsync.CallbackHandler {
		
	    private ConcurrentMap<ContainerId, Container> containers = new ConcurrentHashMap<ContainerId, Container>();
	    private final MOHA_Manager applicationMaster;

	    public NMCallbackHandler(MOHA_Manager applicationMaster) {
	      this.applicationMaster = applicationMaster;
	    }

	    public void addContainer(ContainerId containerId, Container container) {
	      containers.putIfAbsent(containerId, container);
	    }

	    public void onContainerStopped(ContainerId containerId) {
	      LOG.debug("Succeeded to stop Container {}", containerId);	      
	      containers.remove(containerId);
	    }

	    public void onContainerStatusReceived(ContainerId containerId, ContainerStatus containerStatus) {
	      LOG.debug("Container Status: id = {}, status = {}", containerId, containerStatus);
	    }

	    public void onContainerStarted(ContainerId containerId, Map<String, ByteBuffer> allServiceResponse) {
	      LOG.debug("Succeeded to start Container {}", containerId);
	      Container container = containers.get(containerId);
	      if (container != null) {
	        applicationMaster.nmClient.getContainerStatusAsync(containerId, container.getNodeId());
	      }
	    }

	    public void onStartContainerError(ContainerId containerId, Throwable t) {
	      LOG.error("Failed to start Container {}", containerId);
	      containers.remove(containerId);
	      applicationMaster.numCompletedContainers.incrementAndGet();
	    }

	    public void onGetContainerStatusError(ContainerId containerId, Throwable t) {
	      LOG.error("Failed to query the status of Container {}", containerId);
	    }

	    public void onStopContainerError(ContainerId containerId, Throwable t) {
	      LOG.error("Failed to stop Container {}", containerId);
	      containers.remove(containerId);
	    }
	  }//The end of NMCallbackHandler class

}//The end of MOHA_Manager class
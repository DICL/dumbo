package org.kisti.moha;

import java.io.IOException;
import java.net.InetAddress;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Vector;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.yarn.api.ApplicationConstants.Environment;
import org.apache.hadoop.yarn.api.protocolrecords.RegisterApplicationMasterResponse;
import org.apache.hadoop.yarn.api.records.Container;
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

public class MOHA_Manager {
	public class RMCallbackHandler implements CallbackHandler {

		@Override
		public void onContainersCompleted(List<ContainerStatus> statuses) {
			// TODO Auto-generated method stub

			for (ContainerStatus status : statuses) {
				numCompletedContainers.incrementAndGet();
				// LOG.debug(status.toString());
			}
		}

		@Override
		public void onContainersAllocated(List<Container> containers) {
			// TODO Auto-generated method stub

			for (Container container : containers) {
				//LOG.all(container.toString());
				ContainerLauncher launcher = new ContainerLauncher(container, numAllocatedContainers.getAndIncrement(), containerListener);
				Thread mhmThread = new Thread(launcher);
				mhmThread.start();
				launchThreads.add(mhmThread);

			}
		}

		@Override
		public void onShutdownRequest() {
			// TODO Auto-generated method stub
			done = true;
		}

		@Override
		public void onNodesUpdated(List<NodeReport> updatedNodes) {
			// TODO Auto-generated method stub

		}

		@Override
		public float getProgress() {
			// TODO Auto-generated method stub
			float progress = numOfContainers <= 0 ? 0 : (float) numCompletedContainers.get() / numOfContainers;
			return progress;
		}

		@Override
		public void onError(Throwable e) {
			// TODO Auto-generated method stub
			done = true;
			amRMClient.stop();
		}

	}

	private YarnConfiguration conf;
	private AMRMClientAsync<ContainerRequest> amRMClient;

	private FileSystem fileSystem;
	private int numOfContainers;
	protected AtomicInteger numCompletedContainers = new AtomicInteger();
	protected AtomicInteger numAllocatedContainers = new AtomicInteger();
	private volatile boolean done;
	protected NMClientAsync nmClient;
	private NMCallbackHandler containerListener;
	private List<Thread> launchThreads = new ArrayList<>();

	private static MOHA_Logger LOG;
	// private static String inputQueueName;

	private MOHA_Info appInfo;
	private MOHA_Database db;

	Vector<CharSequence> statistic = new Vector<>(30);

	public MOHA_Manager(String[] args) throws IOException {

		setAppInfo(new MOHA_Info());
		getAppInfo().setAppId(args[0]);
		getAppInfo().setExecutorMemory(Integer.parseInt(args[1]));
		getAppInfo().setNumExecutors(Integer.parseInt(args[2]));
		getAppInfo().setNumPartitions(getAppInfo().getNumExecutors());
		getAppInfo().setStartingTime(Long.parseLong(args[3]));
		getAppInfo().setQueueName(getAppInfo().getAppId());

		String zookeeperConnect = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_CONNECT);
		String bootstrapServer = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER);

		LOG = new MOHA_Logger(MOHA_Manager.class, Boolean.parseBoolean(getAppInfo().getConf().getKafkaDebugEnable()),
				getAppInfo().getConf().getDebugQueueName(), zookeeperConnect, bootstrapServer, getAppInfo().getAppId(), 1005);

		LOG.register();

		LOG.all("MOHA_Manager is started in " + InetAddress.getLocalHost().getHostAddress());

		conf = new YarnConfiguration();
		fileSystem = FileSystem.get(conf);

		LOG.debug(conf.toString());
		LOG.debug(fileSystem.toString());

		for (int i = 0; i < args.length; i++) {
			String str = args[i];
			LOG.debug("args[" + i + "] : " + str);
		}

		LOG.all(getAppInfo().toString());
		db = new MOHA_Database(Boolean.parseBoolean(getAppInfo().getConf().getMysqlLogEnable()));
		LOG.all(db.toString());

	}

	private void run() throws YarnException, IOException {

		MOHA_Zookeeper zks = new MOHA_Zookeeper(MOHA_Manager.class, MOHA_Properties.ZOOKEEPER_ROOT, getAppInfo().getAppId());
		LOG.all(zks.toString());

		amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, new RMCallbackHandler());
		amRMClient.init(conf);
		amRMClient.start();

		LOG.debug("amRMClient is started:" + amRMClient.toString());

		RegisterApplicationMasterResponse response;
		response = amRMClient.registerApplicationMaster(NetUtils.getHostname(), -1, "");
		LOG.debug("RegisterApplicationMasterResponse : " + response.toString());

		containerListener = new NMCallbackHandler(this);
		nmClient = NMClientAsync.createNMClientAsync(containerListener);
		nmClient.init(conf);
		nmClient.start();
		LOG.debug("nmClient is started : " + nmClient.toString());

		getAppInfo().setAllocationTime(System.currentTimeMillis());
		// Request resources to launch containers
		Resource capacity = Records.newRecord(Resource.class);
		capacity.setMemory(getAppInfo().getExecutorMemory());
		capacity.setVirtualCores(1);
		Priority pri = Records.newRecord(Priority.class);
		pri.setPriority(0);

		zks.setTimming(MOHA_Properties.TIMMING_INIT);
		zks.setPollingEnable(false);

		for (int i = 0; i < getAppInfo().getNumExecutors(); i++) {

			ContainerRequest containerRequest = new ContainerRequest(capacity, null, null, pri);
			amRMClient.addContainerRequest(containerRequest);
			numOfContainers++;

			LOG.debug("ContainerRequest" + containerRequest.toString());
			LOG.debug("AMRMClientAsync.createAMRMClientAsync: " + amRMClient.toString());
		}
		long start_time = System.currentTimeMillis();
		int numRunning = zks.getNumExecutorsRunning();
		// Wait until all TaskExecutors are launched
		while (!done && (zks.getNumExecutorsRunning() < numOfContainers)) {
			if(zks.getNumExecutorsRunning() != numRunning){
				numRunning = zks.getNumExecutorsRunning();
				LOG.debug("Number of Task Executors running: " + numRunning + "/" + numOfContainers);				
			}
			
			zks.setManagerRunning(true);
			zks.setSystemTime(System.currentTimeMillis() - start_time);
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if (zks.getStopRequest()) {
				LOG.debug("Time out");
				break;
			}
		}
		zks.setStopRequest(false);
		// Allow Task Executors to poll tasks for job queue and execute them
		zks.setPollingEnable(true);
		LOG.debug("Number of Task Executors running: " + zks.getNumExecutorsRunning() + "/" + numOfContainers);
		int numCompleted = numCompletedContainers.get();
		long startingTime = System.currentTimeMillis();
		while (!zks.getStopRequest()) {
			if (numCompletedContainers.get() != numCompleted) {
				numCompleted = numCompletedContainers.get();
				LOG.info("There are " + numCompletedContainers.get() + " per " + numOfContainers + " TaskExecutors have finished");
			}
			zks.setSystemTime(System.currentTimeMillis() - start_time);
			zks.setManagerRunning(true);// send heart beat to MOHA Client
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if((System.currentTimeMillis() - startingTime) > MOHA_Properties.SESSION_MAXTIME_TIMEOUT){
				LOG.debug("Timeout");
				zks.setStopRequest(true);
				break;
			}
			//LOG.info("working -------------------------------------");
		}
		
		//LOG.info("out -------------------------------------");

		// Wait for completing Task Executors
		try {
			Thread.sleep(5000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		LOG.info("There are " + numCompletedContainers.get() + " per " + numOfContainers + " TaskExecutors have finished");

		getAppInfo().setMakespan(System.currentTimeMillis() - getAppInfo().getStartingTime());

		LOG.debug(getAppInfo().toString());
		LOG.all("Application complete! Stop AMRMClientAsync and NMClientAsync, and unregisterApplicationMaster");

		db.insertAppInfoToDababase(getAppInfo());
		zks.setResultsApp(getAppInfo());
		nmClient.stop();
		amRMClient.unregisterApplicationMaster(FinalApplicationStatus.SUCCEEDED, "Application complete!", null);
		amRMClient.stop();

		zks.close();

	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try {
			MOHA_Manager mhm = new MOHA_Manager(args);
			mhm.run();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (YarnException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public MOHA_Info getAppInfo() {
		return appInfo;
	}

	public void setAppInfo(MOHA_Info appInfo) {
		this.appInfo = appInfo;
	}

	protected class ContainerLauncher implements Runnable {
		private Container container;
		@SuppressWarnings("unused")
		private NMCallbackHandler containerListener;
		private int id;

		public ContainerLauncher(Container container, int id, NMCallbackHandler containerListener) {
			super();
			this.container = container;
			this.containerListener = containerListener;
			this.id = id;
			// LOG.debug(this.toString());
		}

		private String getLaunchCommand(Container container, int id) {

			Vector<CharSequence> vargs = new Vector<>(30);
			vargs.add(Environment.JAVA_HOME.$() + "/bin/java");
			vargs.add(MOHA_TaskExecutor.class.getName());

			// add parameters
			vargs.add(getAppInfo().getAppId());
			vargs.add(container.getId().toString());
			vargs.add(String.valueOf(id));

			vargs.add("1><LOG_DIR>/MOHA_TaskExecutor.stdout");
			vargs.add("2><LOG_DIR>/MOHA_TaskExecutor.stderr");
			StringBuilder command = new StringBuilder();
			for (CharSequence str : vargs) {
				command.append(str).append(" ");
			}

			return command.toString();
		}

		@Override
		public void run() {

			Map<String, LocalResource> localResources = new HashMap<>();
			Map<String, String> env = System.getenv();
			LocalResource appJarFile = Records.newRecord(LocalResource.class);
			appJarFile.setType(LocalResourceType.FILE);
			appJarFile.setVisibility(LocalResourceVisibility.APPLICATION);
			try {
				appJarFile.setResource(ConverterUtils.getYarnUrlFromURI(new URI(env.get(MOHA_Properties.APP_JAR))));
			} catch (URISyntaxException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			appJarFile.setTimestamp(Long.valueOf((env.get(MOHA_Properties.APP_JAR_TIMESTAMP))));
			appJarFile.setSize(Long.valueOf(env.get(MOHA_Properties.APP_JAR_SIZE)));

			//LOG.debug(appJarFile.toString());
			localResources.put("app.jar", appJarFile);

			// Setting moha dependencies package
			String appType = System.getenv(MOHA_Properties.APP_TYPE);
			if(appType.equals("S")){
				LocalResource mohaPackage = Records.newRecord(LocalResource.class);
				mohaPackage.setType(LocalResourceType.ARCHIVE);
				mohaPackage.setVisibility(LocalResourceVisibility.APPLICATION);
				try {
					mohaPackage.setResource(ConverterUtils.getYarnUrlFromURI(new URI(env.get(MOHA_Properties.MOHA_TGZ))));
				} catch (URISyntaxException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
				mohaPackage.setTimestamp(Long.valueOf((env.get(MOHA_Properties.MOHA_TGZ_TIMESTAMP))));
				mohaPackage.setSize(Long.valueOf(env.get(MOHA_Properties.MOHA_TGZ_SIZE)));
				localResources.put(MOHA_Properties.EXECUTABLE_DIR, mohaPackage);				
			}


			ContainerLaunchContext context = Records.newRecord(ContainerLaunchContext.class);
			context.setEnvironment(env);
			context.setLocalResources(localResources);

			String command = getLaunchCommand(container, this.id);
			//LOG.all("command = " + command);
			List<String> commands = new ArrayList<>();
			commands.add(command);
			context.setCommands(commands);

			nmClient.startContainerAsync(container, context);
			//LOG.debug(container.toString());
			//LOG.debug(context.toString());
			//LOG.debug(nmClient.toString());
		}
	}
}

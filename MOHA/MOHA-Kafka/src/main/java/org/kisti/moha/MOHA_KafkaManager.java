package org.kisti.moha;

import java.io.IOException;
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

public class MOHA_KafkaManager {
	public class RMCallbackHandler implements CallbackHandler {

		@Override
		public void onContainersCompleted(List<ContainerStatus> statuses) {
			// TODO Auto-generated method stub

			for (ContainerStatus status : statuses) {
				numCompletedContainers.incrementAndGet();

			}
		}

		@Override
		public void onContainersAllocated(List<Container> containers) {
			// TODO Auto-generated method stub

			for (Container container : containers) {

				ContainerLauncher launcher = new ContainerLauncher(container, numAllocatedContainers.getAndIncrement(),
						containerListener);
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
	private KBCallbackHandler containerListener;
	private List<Thread> launchThreads = new ArrayList<>();

	private MOHA_Logger debugLogger;
	private MOHA_Zookeeper zkconnect;

	private MOHA_KafkaInfo kafkaInfo;
	Vector<CharSequence> statistic = new Vector<>(30);

	public MOHA_KafkaManager(String[] args) throws IOException {

		conf = new YarnConfiguration();
		fileSystem = FileSystem.get(conf);
		for (String str : args) {

		}

		setKafkaInfo(new MOHA_KafkaInfo());

		getKafkaInfo().setAppId(args[0]);
		getKafkaInfo().setBrokerMem(Integer.parseInt(args[1]));
		getKafkaInfo().setNumBrokers(Integer.parseInt(args[2]));
		getKafkaInfo().setStartingTime(Long.parseLong(args[3]));

		getKafkaInfo().setNumPartitions(getKafkaInfo().getNumBrokers());

		debugLogger = new MOHA_Logger(MOHA_KafkaManager.class,
				Boolean.parseBoolean(getKafkaInfo().getConf().getKafkaDebugEnable()),
				getKafkaInfo().getConf().getDebugQueueName());

		zkconnect = new MOHA_Zookeeper(MOHA_Manager.class, MOHA_Properties.ZOOKEEPER_ROOT_KAFKA,
				getKafkaInfo().getAppId());

		debugLogger.debug(getKafkaInfo().getConf().toString());
		debugLogger.debug("MOHA_KafkaManager");
		debugLogger.debug(getKafkaInfo().toString());

	}

	public void run() throws YarnException, IOException {

		amRMClient = AMRMClientAsync.createAMRMClientAsync(1000, new RMCallbackHandler());
		amRMClient.init(conf);

		amRMClient.start();
		RegisterApplicationMasterResponse response;
		response = amRMClient.registerApplicationMaster(NetUtils.getHostname(), -1, "");

		containerListener = new KBCallbackHandler(this);
		nmClient = NMClientAsync.createNMClientAsync(containerListener);
		nmClient.init(conf);
		nmClient.start();

		getKafkaInfo().setAllocationTime(System.currentTimeMillis());
		// request resources to launch containers
		Resource capacity = Records.newRecord(Resource.class);
		capacity.setMemory(getKafkaInfo().getBrokerMem());
		Priority pri = Records.newRecord(Priority.class);
		pri.setPriority(0);

		for (int i = 0; i < getKafkaInfo().getNumBrokers(); i++) {

			ContainerRequest containerRequest = new ContainerRequest(capacity, null, null, pri);
			amRMClient.addContainerRequest(containerRequest);
			numOfContainers++;
		}

		try {
			Thread.sleep(1000);

		} catch (InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		while (!done && (numCompletedContainers.get() < numOfContainers)) {

			try {
				// Send heart beat to client
				zkconnect.setStatus(true);
				Thread.sleep(1000);

			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		getKafkaInfo().setMakespan(System.currentTimeMillis() - getKafkaInfo().getStartingTime());
		debugLogger.debug("setMakespan");

		debugLogger.debug(getKafkaInfo().getConf().getKafkaClusterId());
		nmClient.stop();
		debugLogger.debug("stop");
		amRMClient.unregisterApplicationMaster(FinalApplicationStatus.SUCCEEDED, "Application complete!", null);
		debugLogger.debug("unregisterApplicationMaster");
		amRMClient.stop();
		debugLogger.debug("stop");

	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		try {
			MOHA_KafkaManager mhm = new MOHA_KafkaManager(args);
			mhm.run();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (YarnException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public MOHA_KafkaInfo getKafkaInfo() {
		return kafkaInfo;
	}

	public void setKafkaInfo(MOHA_KafkaInfo kafkaInfo) {
		this.kafkaInfo = kafkaInfo;
	}

	protected class ContainerLauncher implements Runnable {
		private Container container;
		@SuppressWarnings("unused")
		private KBCallbackHandler containerListener;
		private int id;

		public ContainerLauncher(Container container, int id, KBCallbackHandler containerListener) {
			super();
			this.container = container;
			this.containerListener = containerListener;
			this.id = id;

		}

		public String getLaunchCommand(Container container, int containerId) {
			Vector<CharSequence> vargs = new Vector<>(30);
			vargs.add(Environment.JAVA_HOME.$() + "/bin/java");
			vargs.add(MOHA_KafkaBrokerLauncher.class.getName());
			// set broker id
			vargs.add(container.getId().toString());
			vargs.add(String.valueOf(containerId));
			vargs.add(getKafkaInfo().getAppId());

			vargs.add("1><LOG_DIR>/MOHA_KafkaBrokerLauncher.stdout");
			vargs.add("2><LOG_DIR>/MOHA_KafkaBrokerLauncher.stderr");
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
			localResources.put("app.jar", appJarFile);

			// Setting Kafka dependencies package
			LocalResource kafkaPackage = Records.newRecord(LocalResource.class);
			kafkaPackage.setType(LocalResourceType.ARCHIVE);
			kafkaPackage.setVisibility(LocalResourceVisibility.APPLICATION);
			try {
				kafkaPackage.setResource(ConverterUtils.getYarnUrlFromURI(new URI(env.get(MOHA_Properties.KAFKA_TGZ))));
			} catch (URISyntaxException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			kafkaPackage.setTimestamp(Long.valueOf((env.get(MOHA_Properties.KAFKA_TGZ_TIMESTAMP))));
			kafkaPackage.setSize(Long.valueOf(env.get(MOHA_Properties.KAFKA_TGZ_SIZE)));
			localResources.put("kafkaLibs", kafkaPackage);

			ContainerLaunchContext context = Records.newRecord(ContainerLaunchContext.class);
			context.setEnvironment(env);
			context.setLocalResources(localResources);

			String command = getLaunchCommand(container, this.id);
			List<String> commands = new ArrayList<>();
			commands.add(command);
			context.setCommands(commands);

			nmClient.startContainerAsync(container, context);

		}
	}

}

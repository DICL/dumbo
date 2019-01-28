package org.kisti.moha;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.net.NetUtils;
import org.apache.zookeeper.KeeperException;

public class MOHA_KafkaBrokerLauncher {

	private MOHA_KafkaBrokerInfo kbInfo;
	private static MOHA_Logger LOG;
	private static MOHA_Zookeeper zkconnect;

	public MOHA_KafkaBrokerLauncher(String[] args) {
		kbInfo = new MOHA_KafkaBrokerInfo();
		
		kbInfo.setContainerId(args[0]);
		kbInfo.setBrokerId(args[1]);
		kbInfo.setAppId(args[2]);
		kbInfo.setHostname(NetUtils.getHostname());

		kbInfo.setKafkaBinDir(kbInfo.getConf().getKafkaVersion());
		kbInfo.setZookeeperConnect(kbInfo.getConf().getZookeeperConnect() + "/" + kbInfo.getConf().getKafkaClusterId());

		LOG = new MOHA_Logger(MOHA_KafkaBrokerLauncher.class,
				(Boolean.parseBoolean(kbInfo.getConf().getKafkaDebugEnable())), kbInfo.getConf().getDebugQueueName(),kbInfo.getAppId(),Integer.valueOf(kbInfo.getBrokerId()));
		
		LOG.debug("MOHA_KafkaBrokerLauncher [" + kbInfo.getBrokerId() + "] is started in " + kbInfo.getHostname());

		zkconnect = new MOHA_Zookeeper(MOHA_KafkaBrokerLauncher.class, MOHA_Properties.ZOOKEEPER_ROOT,
				kbInfo.getConf().getKafkaClusterId());

		if (zkconnect == null) {
			LOG.debug("zkconnect error");
			return;
		}	

	}

	public void run() throws IOException, InterruptedException, KeeperException {

		BrokerProperties props = new BrokerProperties();
		List<Integer> preIds;
		List<Integer> brokerIds;

		preIds = getRunningBrokerProcessIDs();
		for (int i = 0; i < preIds.size(); i++) {
			LOG.debug("MOHA_KafkaBrokerLauncher [" + kbInfo.getBrokerId() + "] preIds:" + preIds.get(i).toString());
		}
		stopBroker(preIds);		
		
		Thread.sleep(6000);
		
		props.setBrokerId(kbInfo.getBrokerId());
		props.setListeners(getOpenPort());
		props.setLogDirs(MOHA_Properties.KAFKA_LOG_DIR);
		props.setZookeeperConnect(MOHA_Properties.ZOOKEEPER_ROOT + "/" + kbInfo.getConf().getKafkaClusterId());

		LOG.debug(props.toString());

		ThreadStartBroker startBroker = new ThreadStartBroker(props);
		Thread startThread = new Thread(startBroker);

		startThread.start();
		LOG.debug("startThread.start()");
		try {
			Thread.sleep(10000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		brokerIds = getStatedBrokerIds(preIds);
		if (!zkconnect.isBrokerStated()) {
			LOG.debug(MOHA_Common.convertLongToDate(System.currentTimeMillis())
					+ "  Kafka broker crashed or haven't worked yet ");
			Thread.sleep(1000);
		}

		zkconnect.setStopRequest(false);
		LOG.debug(MOHA_Common.convertLongToDate(System.currentTimeMillis()) + "  Kafka brokers are running on "
				+ zkconnect.getBootstrapServers());

		
		while (!zkconnect.getStopRequest()) {
			try {
				Thread.sleep(1000);
				// setRequests(true);
			} catch (InterruptedException e) { // TODO
				// Auto-generated catch block e.printStackTrace();
			}
		}

		stopBroker(brokerIds);
		LOG.debug("MOHA_KafkaBrokerLauncher [" + kbInfo.getBrokerId() + "] is stopped ");

		zkconnect.close();
		return;
	};

	public static void main(String[] args) throws KeeperException, IOException {

		
		for (String arg:args){
			System.out.println(arg);
		}
		MOHA_KafkaBrokerLauncher brokerLauncher = new MOHA_KafkaBrokerLauncher(args);

		try {
			brokerLauncher.run();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return;
	}

	/**
	 * @return
	 * @throws IOException
	 * @throws InterruptedException
	 */

	private List<Integer> getRunningBrokerProcessIDs() throws IOException, InterruptedException {
		List<Integer> processIds = new ArrayList<>();

		Path shell = FileSystems.getDefault().getPath("getIds.sh");
		if (Files.exists(shell)) {
			Files.delete(shell);
		}

		Set<PosixFilePermission> perms = PosixFilePermissions.fromString("rwxr-x---");
		FileAttribute<Set<PosixFilePermission>> attrs = PosixFilePermissions.asFileAttribute(perms);
		Files.createFile(shell, attrs);

		List<String> lines = Arrays.asList("ps ax | grep -i 'kafka_' | grep java | grep -v grep | awk '{print $1}'");

		Files.write(shell, lines, Charset.forName("UTF-8"));

		ProcessBuilder builder = new ProcessBuilder("./getIds.sh");
		Process p;
		String line;
		p = builder.start();
		p.waitFor();
		BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
		while ((line = br.readLine()) != null) {
			processIds.add(Integer.parseInt(line));
		}
		Files.delete(shell);

		return processIds;
	}

	private List<Integer> getStatedBrokerIds(List<Integer> preIds) throws IOException, InterruptedException {
		List<Integer> currentIds;
		List<Integer> brokerIds = new ArrayList<>();
		currentIds = getRunningBrokerProcessIDs();

		for (int i = 0; i < currentIds.size(); i++) {

			boolean found = true;
			for (int j = 0; j < preIds.size(); j++) {

				if (preIds.get(j).intValue() == currentIds.get(i).intValue()) {

					found = false;
				}
			}

			if (found) {
				brokerIds.add(currentIds.get(i).intValue());

				LOG.debug("Broker Id = " + currentIds.get(i).intValue());
			} else {
				LOG.debug("Background appIds = " + currentIds.get(i).toString());
			}

		}

		return brokerIds;
	}

	private int getOpenPort() {

		int port = 9092;
		boolean open = true;
		while (true) {
			List<String> command = new ArrayList<String>();
			command.add("netstat");
			command.add("-lntu");

			ProcessBuilder builder = new ProcessBuilder(command);
			Process p;
			String line;
			try {
				p = builder.start();
				p.waitFor();
				BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
				while ((line = br.readLine()) != null) {
					// LOG.info(line);
					if (line.contains(Integer.toString(port))) {
						open = false;
					}
					;
				}
				if (open) {

					return port;
				}
				open = true;
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			port+=3;
		}
	}

	private void startBroker() {

		List<String> command = new ArrayList<String>();
		command.add(kbInfo.getKafkaBinDir() + "/bin/kafka-server-start.sh");
		command.add(kbInfo.getKafkaBinDir() + "/config/" + MOHA_Properties.KAFKA_SERVER_PROPERTIES);

		ProcessBuilder builder = new ProcessBuilder(command);
		Process p;
		String line;
		LOG.debug("ProcessBuilder builder = " + command);
		try {

			p = builder.start();
			LOG.debug("p = builder.start();");
//			try {
//				p.waitFor();
//			} catch (InterruptedException e) {
//				// TODO Auto-generated catch block
//				e.printStackTrace();
//			}

//			Thread.sleep(2000);
//			LOG.debug("p.waitFor()");
			BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
			while ((line = br.readLine()) != null) {
				LOG.debug(line);
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();		
		}
	}

	private void stopBroker(List<Integer> appId) throws IOException, InterruptedException {
		Runtime rt = Runtime.getRuntime();
		for(int i =0;i<appId.size();i++){
			LOG.debug("kill -9 " + String.valueOf(appId.get(i).intValue()));
			rt.exec("kill -9 " + appId.get(i).intValue());
		}
		
	}

	private void configBrokerProperties(BrokerProperties props) throws IOException {
		BufferedWriter out;
		BufferedReader br;
		ArrayList<String> lines;
		LOG.debug(props.toString());
		lines = new ArrayList<String>();
		String line = null;

		try {

			File serverProperties = new File(kbInfo.getKafkaBinDir() + "/config/server.properties");
			File brokerProperties = new File(
					kbInfo.getKafkaBinDir() + "/config/" + MOHA_Properties.KAFKA_SERVER_PROPERTIES);
			serverProperties.setWritable(true);
			FileReader fr = new FileReader(serverProperties);
			br = new BufferedReader(fr);
			line = br.readLine();
			while (line != null) {
				if (line.contains("broker.id=0")) {
					line = line.replace("broker.id=0", "broker.id=" + props.getBrokerId());
					LOG.debug(line);
				}

				if (line.contains("#listeners=PLAINTEXT://:9092")) {
					line = line.replace("#listeners=PLAINTEXT://:9092",
							"listeners=PLAINTEXT://:" + Integer.toString(props.getListeners()));

					LOG.debug(line);
				}
				//version 0.9.0.1
//				if (line.contains("listeners=PLAINTEXT://:9092")) {
//					line = line.replace("listeners=PLAINTEXT://:9092",
//							"listeners=PLAINTEXT://:" + Integer.toString(props.getListeners()));
//
//					LOG.all(line);
//				}

				if (line.contains("log.dirs=/tmp/kafka-logs")) {
					line = line.replace("log.dirs=/tmp/kafka-logs", "log.dirs=" + props.getLogDirs());

					LOG.debug(line);
				}
				//version 0.9.0.1
//				if (line.contains("log.dirs=/tmp/kafka-logs")) {
//					line = line.replace("log.dirs=/tmp/kafka-logs", "log.dirs=" + props.getLogDirs());
//
//					LOG.all(line);
//				}

				if (line.contains("zookeeper.connect=localhost:2181")) {
					line = line.replace("zookeeper.connect=localhost:2181",
							"zookeeper.connect=localhost:2181/" + props.getZookeeperConnect());

					LOG.debug(line);
				}
				
				if (line.contains("#delete.topic.enable=true")) {
					line = line.replace("#delete.topic.enable=true", "delete.topic.enable=true");

					LOG.debug(line);
				}

				lines.add(line);
				line = br.readLine();
			}
			lines.add("group.max.session.timeout.ms=3600000");
			fr.close();
			br.close();
			LOG.debug("start writing broker properties");
			FileWriter fw = new FileWriter(brokerProperties);
			out = new BufferedWriter(fw);
			for (String s : lines)
				out.write(s + '\n');

			out.flush();
			out.close();
			LOG.debug("end writing broker properties");
		} catch (Exception ex) {
			ex.printStackTrace();
		}
	}

	protected class ThreadStartBroker implements Runnable {
		private BrokerProperties props;

		public ThreadStartBroker(BrokerProperties props) {
			// TODO Auto-generated constructor stub
			this.props = props;

		}

		@Override
		public void run() {
			// TODO Auto-generated method stub

			try {
				configBrokerProperties(props);
				startBroker();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			return;
		}

	}

	private class BrokerProperties {
		String brokerId;
		int listeners;
		String logDirs;
		String zookeeperConnect;

		@Override
		public String toString() {
			return "KafkaServerProperties [brokerId=" + brokerId + ", listeners=" + listeners + ", logDirs=" + logDirs
					+ ", zookeeperConnect=" + zookeeperConnect + "]";
		}

		public String getBrokerId() {
			return brokerId;
		}

		public void setBrokerId(String brokerId) {
			this.brokerId = brokerId;
		}

		public int getListeners() {
			return listeners;
		}

		public void setListeners(int listeners) {
			this.listeners = listeners;
		}

		public String getLogDirs() {
			return logDirs;
		}

		public void setLogDirs(String logDirs) {
			this.logDirs = logDirs;
		}

		public String getZookeeperConnect() {
			return zookeeperConnect;
		}

		public void setZookeeperConnect(String zookeeperConnect) {
			this.zookeeperConnect = zookeeperConnect;
		}
	}
}

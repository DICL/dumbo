package org.kisti.moha;

public class MOHA_KafkaBrokerInfo {

	private String appId;
	public String getAppId() {
		return appId;
	}

	public void setAppId(String appId) {
		this.appId = appId;
	}

	private String brokerId;
	private String containerId;
	private String zookeeperConnect;
	private String kafkaBinDir;
	private int port;
	private String hostname;

	private MOHA_Configuration conf;

	public MOHA_KafkaBrokerInfo() {
		conf = new MOHA_Configuration(System.getenv());
	}

	public MOHA_Configuration getConf() {
		return conf;
	}

	public void setConf(MOHA_Configuration conf) {
		this.conf = conf;
	}

	private long launchedTime;

	public String getKafkaBinDir() {
		return kafkaBinDir;
	}

	public void setKafkaBinDir(String kafkaLibVersion) {
		this.kafkaBinDir = "kafkaLibs/" + kafkaLibVersion;
	}

	@Override
	public String toString() {
		return "MOHA_KafkaBrokerInfo [brokerId=" + brokerId + ", containerId=" + containerId + ", zookeeperConnect="
				+ zookeeperConnect + ", kafkaBinDir=" + kafkaBinDir + ", port=" + port + ", hostname=" + hostname
				+ ", conf=" + conf + ", launchedTime=" + launchedTime + "]";
	}

	public String getContainerId() {
		return containerId;
	}

	public void setContainerId(String containerId) {
		this.containerId = containerId;
	}

	public String getHostname() {
		return hostname;
	}

	public void setHostname(String hostname) {
		this.hostname = hostname;
	}

	public long getLaunchedTime() {
		return launchedTime;
	}

	public void setLaunchedTime(long launchedTime) {
		this.launchedTime = launchedTime;
	}

	public String getBrokerId() {
		return brokerId;
	}

	public void setBrokerId(String brokerId) {
		this.brokerId = brokerId;
	}

	public int getPort() {
		return port;
	}

	public void setPort(int port) {
		this.port = port;
	}

	public String getZookeeperConnect() {
		return zookeeperConnect;
	}

	public void setZookeeperConnect(String zookeeperConnect) {
		this.zookeeperConnect = zookeeperConnect;
	}

}

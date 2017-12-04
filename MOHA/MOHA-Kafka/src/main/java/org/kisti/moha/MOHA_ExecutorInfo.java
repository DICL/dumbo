package org.kisti.moha;

public class MOHA_ExecutorInfo {
	private String appId;
	private int executorId;
	private String containerId;
	private long firstMessageTime;
	private String hostname;
	private long launchedTime;
	private int numExecutedTasks;
	private long executionTime;
	private int pollingTime;
	private long endingTime;
	private String queueName;
	private int pushingTime;
	private long pushingRate;
	
	public long getPushingRate() {
		return pushingRate;
	}

	public void setPushingRate(long pushingRate) {
		this.pushingRate = pushingRate;
	}

	public long getPollingRate() {
		return pollingRate;
	}

	public void setPollingRate(long pollingRate) {
		this.pollingRate = pollingRate;
	}

	private long pollingRate;
	public int getPushingTime() {
		return pushingTime;
	}

	public void setPushingTime(int pushingTime) {
		this.pushingTime = pushingTime;
	}

	public void setPollingTime(int pollingTime) {
		this.pollingTime = pollingTime;
	}

	MOHA_Configuration conf;

	public MOHA_ExecutorInfo() {
		conf = new MOHA_Configuration(System.getenv());
	}

	@Override
	public String toString() {
		return "MOHA_ExecutorInfo [appId=" + appId + ", executorId=" + executorId + ", containerId=" + containerId
				+ ", firstMessageTime=" + firstMessageTime + ", hostname=" + hostname + ", launchedTime=" + launchedTime
				+ ", numExecutedTasks=" + numExecutedTasks + ", runningTime=" + executionTime + ", pollingTime="
				+ pollingTime + ", endingTime=" + endingTime + ", queueName=" + queueName + ", conf=" + conf + "]";
	}

	public MOHA_Configuration getConf() {
		return conf;
	}

	public void setConf(MOHA_Configuration conf) {
		this.conf = conf;
	}

	public long getEndingTime() {
		return endingTime;
	}

	public void setEndingTime(long endingTime) {
		this.endingTime = endingTime;
	}

	public long getFirstMessageTime() {
		return firstMessageTime;
	}

	public void setFirstMessageTime(long firstMessageTime) {
		if (this.firstMessageTime == 0) {
			this.firstMessageTime = firstMessageTime;
		}

	}

	public int getPollingTime() {
		return pollingTime;
	}

	public void setNumOfPolls(int pollingTime) {
		this.pollingTime = pollingTime;
	}

	public String getContainerId() {
		return containerId;
	}

	public void setContainerId(String containerId) {
		this.containerId = containerId;
	}

	public String getAppId() {
		return appId;
	}

	public void setAppId(String appId) {
		this.appId = appId;
	}



	public int getExecutorId() {
		return executorId;
	}

	public void setExecutorId(int executorId) {
		this.executorId = executorId;
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

	public int getNumExecutedTasks() {
		return numExecutedTasks;
	}

	public void setNumExecutedTasks(int numExecutedTasks) {
		this.numExecutedTasks = numExecutedTasks;
	}

	public long getExecutionTime() {
		return executionTime;
	}

	public void setExecutionTime(long exitTime) {
		this.executionTime = exitTime;
	}

	public String getQueueName() {
		return queueName;
	}

	public void setQueueName(String queueName) {
		this.queueName = queueName;
	}

}

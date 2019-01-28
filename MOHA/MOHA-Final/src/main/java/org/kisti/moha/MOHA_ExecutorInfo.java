package org.kisti.moha;

public class MOHA_ExecutorInfo {
	private volatile String appId;
	private volatile int executorId;
	private volatile String containerId;
	private volatile long firstMessageTime;
	private volatile String hostname;
	private volatile long launchedTime;
	private volatile int numExecutedTasks;
	private volatile long executionTime;
	private volatile int pollingTime;
	private volatile long endingTime;
	private volatile String queueName;
	private volatile int pushingTime;
	private volatile long pushingRate;
	private volatile long waitingTime;
	private volatile int numRequestedThreads;
	private volatile int numRunningThreads;
	private volatile int numExecutingTasks;
	private volatile int numSpecifiedTasks;
	private volatile int numActiveThreads;
	private volatile int numQueueFail;
	private volatile boolean isQueueEmpty;
	
	public long getPushingRate() {
		return pushingRate;
	}

	public void setPushingRate(long pushingRate) {
		this.pushingRate = pushingRate;
	}

	public double getPollingRate() {
		return pollingRate;
	}

	public void setPollingRate(long pollingRate) {
		this.pollingRate = pollingRate;
	}

	private double pollingRate;
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

	public int getNumOfPolls() {
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

	public long getWaitingTime() {
		return waitingTime;
	}

	public void setWaitingTime(long waitingTime) {
		this.waitingTime = waitingTime;
	}

	public int getNumRequestedThreads() {
		return numRequestedThreads;
	}

	public void setNumRequestedThreads(int numRequestedThreads) {
		this.numRequestedThreads = numRequestedThreads;
	}

	public int getNumRunningThreads() {
		return numRunningThreads;
	}

	public void setNumRunningThreads(int numRunningThreads) {
		this.numRunningThreads = numRunningThreads;
	}

	public int getNumExecutingTasks() {
		return numExecutingTasks;
	}

	public void setNumExecutingTasks(int numExecutingTasks) {
		this.numExecutingTasks = numExecutingTasks;
	}

	public int getNumSpecifiedTasks() {
		return numSpecifiedTasks;
	}

	public void setNumSpecifiedTasks(int numSpecifiedTasks) {
		this.numSpecifiedTasks = numSpecifiedTasks;
	}

	public boolean isQueueEmpty() {
		return isQueueEmpty;
	}

	public void setQueueEmpty(boolean isQueueEmpty) {
		this.isQueueEmpty = isQueueEmpty;
	}

	public int getNumActiveThreads() {
		return numActiveThreads;
	}

	public void setNumActiveThreads(int numActiveThreads) {
		this.numActiveThreads = numActiveThreads;
	}

	public int getNumQueueFail() {
		return numQueueFail;
	}

	public void setNumQueueFail(int numQueueFail) {
		this.numQueueFail = numQueueFail;
	}

}

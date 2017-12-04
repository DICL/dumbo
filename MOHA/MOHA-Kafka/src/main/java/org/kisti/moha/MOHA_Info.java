package org.kisti.moha;

import org.mortbay.log.Log;

public class MOHA_Info {
	private String appId;
	private String queueName;
	private int executorMemory;
	private int numExecutors;
	private int numPartitions;
	private long startingTime;
	private long makespan;
	private int numCommands;
	private String command;
	private long initTime;
	private long allocationTime;
	private MOHA_Configuration conf;

	public MOHA_Info() {
		conf = new MOHA_Configuration(System.getenv());
	}

	@Override
	public String toString() {
		return "MOHA_Info [appId=" + appId + ", queueName=" + queueName + ", executorMemory=" + executorMemory
				+ ", numExecutors=" + numExecutors + ", numPartitions=" + numPartitions + ", startingTime="
				+ startingTime + ", makespan=" + makespan + ", numCommands=" + numCommands + ", command=" + command
				+ ", initTime=" + initTime + ", allocationTime=" + allocationTime + ", conf=" + conf + "]";
	}

	public MOHA_Configuration getConf() {
		return conf;
	}

	public void setConf(MOHA_Configuration conf) {
		this.conf = conf;
	}

	public long getAllocationTime() {
		return allocationTime;
	}

	public void setAllocationTime(long allocationTime) {
		this.allocationTime = allocationTime;
	}

	public long getInitTime() {
		return initTime;
	}

	public void setInitTime(long l) {
		this.initTime = l;
	}

	public String getAppId() {
		return appId;
	}

	public void setAppId(String appId) {
		Log.info("appId = {}", appId);
		this.appId = appId;
	}

	public int getExecutorMemory() {
		return executorMemory;
	}

	public void setExecutorMemory(int executorMemory) {
		this.executorMemory = executorMemory;
	}

	public int getNumExecutors() {
		return numExecutors;
	}

	public void setNumExecutors(int numExecutors) {
		this.numExecutors = numExecutors;
	}

	public int getNumPartitions() {
		return numPartitions;
	}

	public void setNumPartitions(int numPartitions) {
		this.numPartitions = numPartitions;
	}

	public long getStartingTime() {
		return startingTime;
	}

	public void setStartingTime(long startingTime) {
		this.startingTime = startingTime;
	}

	public long getMakespan() {
		return makespan;
	}

	public void setMakespan(long makespan) {
		this.makespan = makespan;
	}

	public int getNumCommands() {
		return numCommands;
	}

	public void setNumCommands(int numCommands) {
		this.numCommands = numCommands;
	}

	public String getCommand() {
		return command;
	}

	public void setCommand(String command) {
		this.command = command;
	}

	public String getQueueName() {
		return queueName;
	}

	public void setQueueName(String queueName) {
		this.queueName = queueName;
	}
}

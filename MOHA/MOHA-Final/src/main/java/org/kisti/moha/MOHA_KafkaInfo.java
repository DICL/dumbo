package org.kisti.moha;

public class MOHA_KafkaInfo {
	@Override
	public String toString() {
		return "MOHA_KafkaInfo [appId=" + appId + ", brokerMem=" + brokerMem + ", numBrokers=" + numBrokers
				+ ", numPartitions=" + numPartitions + ", startingTime=" + startingTime + ", makespan=" + makespan
				+ ", numCommands=" + numCommands + ", command=" + command + ", initTime=" + initTime
				+ ", allocationTime=" + allocationTime + ", conf=" + conf + "]";
	}

	private String appId;
	private int brokerMem;
	private int numBrokers;
	private int numPartitions;
	private long startingTime;
	private long makespan;
	private int numCommands;
	private String command;
	private long initTime;
	private long allocationTime;
	private MOHA_Configuration conf;

	public MOHA_KafkaInfo() {
		setConf(new MOHA_Configuration(System.getenv()));
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

	public int getBrokerMem() {
		return brokerMem;
	}

	public void setBrokerMem(int executorMemory) {
		this.brokerMem = executorMemory;
	}

	public int getNumBrokers() {
		return numBrokers;
	}

	public void setNumBrokers(int numExecutors) {
		this.numBrokers = numExecutors;
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

	public String getAppId() {
		return appId;
	}

	public void setAppId(String appId) {
		this.appId = appId;
	}

	public MOHA_Configuration getConf() {
		return conf;
	}

	public void setConf(MOHA_Configuration conf) {
		this.conf = conf;
	}

}

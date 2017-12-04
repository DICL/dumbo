package org.kisti.moha;

import java.nio.ByteBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.apache.hadoop.yarn.api.records.Container;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.api.records.ContainerStatus;
import org.apache.hadoop.yarn.client.api.async.NMClientAsync;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NMCallbackHandler implements NMClientAsync.CallbackHandler {

	private static final Logger LOG = LoggerFactory.getLogger(NMCallbackHandler.class);

	private ConcurrentMap<ContainerId, Container> containers = new ConcurrentHashMap<ContainerId, Container>();
	private final MOHA_Manager mohaManager;
	MOHA_Logger debugLogger;

	public NMCallbackHandler(MOHA_Manager applicationMaster) {
		this.mohaManager = applicationMaster;
		debugLogger = new MOHA_Logger(NMCallbackHandler.class,Boolean.parseBoolean(mohaManager.getAppInfo().getConf().getKafkaDebugEnable()),
				mohaManager.getAppInfo().getConf().getDebugQueueName());

		
	}

	public void addContainer(ContainerId containerId, Container container) {
		containers.putIfAbsent(containerId, container);
		
	}

	@Override
	public void onContainerStopped(ContainerId containerId) {
		
		containers.remove(containerId);
	}

	@Override
	public void onContainerStatusReceived(ContainerId containerId, ContainerStatus containerStatus) {
		
	}

	@Override
	public void onContainerStarted(ContainerId containerId, Map<String, ByteBuffer> allServiceResponse) {
				Container container = containers.get(containerId);
		if (container != null) {
		
			mohaManager.nmClient.getContainerStatusAsync(containerId, container.getNodeId());
		}
	}

	@Override
	public void onStartContainerError(ContainerId containerId, Throwable t) {
		
		containers.remove(containerId);
		mohaManager.numCompletedContainers.incrementAndGet();
	}

	@Override
	public void onGetContainerStatusError(ContainerId containerId, Throwable t) {
		
	}

	@Override
	public void onStopContainerError(ContainerId containerId, Throwable t) {
		
		containers.remove(containerId);
	}
}
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

public class KBCallbackHandler implements NMClientAsync.CallbackHandler {

	private static final Logger LOG = LoggerFactory.getLogger(KBCallbackHandler.class);

	private ConcurrentMap<ContainerId, Container> containers = new ConcurrentHashMap<ContainerId, Container>();
	private final MOHA_KafkaManager mohaManager;
	private MOHA_Logger debugLogger;

	public KBCallbackHandler(MOHA_KafkaManager applicationMaster) {
		this.mohaManager = applicationMaster;
		debugLogger = new MOHA_Logger(KBCallbackHandler.class,
				Boolean.parseBoolean(applicationMaster.getKafkaInfo().getConf().getKafkaDebugEnable()),
				applicationMaster.getKafkaInfo().getConf().getDebugQueueName());

		
	}

	public void addContainer(ContainerId containerId, Container container) {
		containers.putIfAbsent(containerId, container);
		
	}

	@Override
	public void onContainerStopped(ContainerId containerId) {
		LOG.debug("Succeeded to stop Container {}", containerId);
		
		containers.remove(containerId);
	}

	@Override
	public void onContainerStatusReceived(ContainerId containerId, ContainerStatus containerStatus) {
		LOG.debug("Container Status: id = {}, status = {}", containerId, containerStatus);
		
	}

	@Override
	public void onContainerStarted(ContainerId containerId, Map<String, ByteBuffer> allServiceResponse) {
		LOG.debug("Succeeded to start Container {}", containerId);

		
		Container container = containers.get(containerId);
		if (container != null) {
			
			mohaManager.nmClient.getContainerStatusAsync(containerId, container.getNodeId());
		}
	}

	@Override
	public void onStartContainerError(ContainerId containerId, Throwable t) {
		
		LOG.error("Failed to start Container {}", containerId);
		containers.remove(containerId);
		mohaManager.numCompletedContainers.incrementAndGet();
	}

	@Override
	public void onGetContainerStatusError(ContainerId containerId, Throwable t) {
		
		LOG.error("Failed to query the status of Container {}", containerId);
	}

	@Override
	public void onStopContainerError(ContainerId containerId, Throwable t) {
		
		LOG.error("Failed to stop Container {}", containerId);
		containers.remove(containerId);
	}
}
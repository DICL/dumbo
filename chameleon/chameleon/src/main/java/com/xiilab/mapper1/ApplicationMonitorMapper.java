package com.xiilab.mapper1;

import java.util.List;
import java.util.Map;

import com.xiilab.models.YarnAppMonitorPerNodeVO;
import com.xiilab.models.YarnAppMonitorVO;

public interface ApplicationMonitorMapper {
	/**
	 * get container id 
	 * @param application_id
	 * @return
	 */
	public List<String> getContainerIdForApplicationMonitor(String application_id);

	/**
	 * get rowkey
	 * @param container_id
	 * @return
	 */
	public String getRowKeyForApplicationMonitor(String container_id);
	
	public List<YarnAppMonitorVO> getYarnAppMonitorListBak(YarnAppMonitorVO yarnAppMonitorVO);
	
	public List<Map<String, Object>> getYarnAppMonitorList(Map<String, Object> find_data);
	
	public String findHyperTable(YarnAppMonitorVO yarnAppMonitorVO);
	
	public List<Map<String, Object>> getYarnAppMonitorListPerNode(Map<String, Object> find_data);
	
	public List<Map<String, Object>> findHyperTablePerNode(YarnAppMonitorPerNodeVO yarnAppMonitorPerNodeVO);

}

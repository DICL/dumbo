package com.xiilab.ambari;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class AmbariStatusServise {
	
	@Autowired
	private AmbariStatusDAO ambariStatusDAO;
	
	public Map<String, String> getHostStatus() {
		return ambariStatusDAO.getHostStatus();
	}

	public Map<String, String> getStorageUsage() {
		return ambariStatusDAO.getStorageUsage();
	}

	public String getyarnAllAppList() {
		return ambariStatusDAO.yarnAppList(null);
	}

	public String getYarnStatus() {
		return ambariStatusDAO.getYarnStatus();
	}

	public String getNodeMatricDataArr() {
		return ambariStatusDAO.getNodeMatricDataArr();
	}

	public String getAlert() {
		return ambariStatusDAO.getAlert();
	}

	public Map<String,Object> getLustreMetrics() {
		return ambariStatusDAO.getLustreMetrics();
	}

//	public String getLustreMetricData(String startTime, String endTime) {
//		return ambariStatusDAO.getLustreMetrics(startTime, endTime);
//	}
	
	public Map<String, Object> getLustreMetricData(String startTime, String endTime) {
		return ambariStatusDAO.getLustreMetrics(startTime, endTime);
	}

	public String getYARNAppMonitorClientNodes() {
		return ambariStatusDAO.getYARNAppMonitorClientNodes();
	}

	public Map<String, Object> new_getLustreMetricData(String metricName, String startTime, String endTime) {
		return ambariStatusDAO.new_getLustreMetricData(metricName,startTime,endTime);
	}
	
	public String getForHostMetricData(String metricName, String startTime, String endTime) {
		return ambariStatusDAO.getForHostMetricData(metricName,startTime,endTime);
	}

	public String getNodeMatricDataArrForHost(String host_name) {
		return ambariStatusDAO.getNodeMatricDataArrForHost(host_name);
	}

	public String getNodeDataArrBak() {
		// TODO Auto-generated method stub
		return ambariStatusDAO.getNodeDataArrBak();
	}
	
}

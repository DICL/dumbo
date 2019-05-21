package com.xiilab.app_monitor;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.xiilab.models.YarnAppMonitorPerNodeVO;
import com.xiilab.models.YarnAppMonitorVO;

@Service
public class ApplicationMonitoringService {
	@Autowired
	private ApplicationMonitoringDAO applicationMonitoringDAO;

	public List<Map<String, Object>> getYarnJobMonitor() {
		return applicationMonitoringDAO.getYarnJobMonitor();
	}

	public List<YarnAppMonitorVO> getYarnJobHistoryBak(YarnAppMonitorVO yarnAppMonitorVO) {
		return applicationMonitoringDAO.getYarnJobHistoryBak(yarnAppMonitorVO);
	}
	
	public List<Map<String, Object>> getYarnJobHistory(YarnAppMonitorVO yarnAppMonitorVO) {
		return applicationMonitoringDAO.getYarnJobHistory(yarnAppMonitorVO);
	}
	
	/**
	 * @author sehunkim
	 * @param yarnAppMonitorVO
	 * @return
	 */
	public List<Map<String, Object>> getYarnJobHistoryPerNode(YarnAppMonitorPerNodeVO yarnAppMonitorPerNodeVO) {
		return applicationMonitoringDAO.getYarnJobHistoryPerNode(yarnAppMonitorPerNodeVO);
	}

	/**
	 *  Metric Registry View 에 저장되어 있는 Metric list 가져오기 
	 * @return
	 */
	public List<Map<String, Object>> getMetricList() {
		return applicationMonitoringDAO.getMetricList();
	}
}

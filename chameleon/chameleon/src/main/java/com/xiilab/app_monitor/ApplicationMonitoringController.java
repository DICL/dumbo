package com.xiilab.app_monitor;

import java.util.List;
import java.util.Map;
import java.util.stream.Collector;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import com.xiilab.models.YarnAppMonitorPerNodeVO;
import com.xiilab.models.YarnAppMonitorVO;

/**
 * YarnAppMonitor & YarnJobhistory 용 컨트롤러
 * @author xiilab
 *
 */
@Controller
public class ApplicationMonitoringController {
	@Autowired
	private ApplicationMonitoringService applicationMonitoringService;
	
	/**
	 * YarnJob 리스트 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYarnJobMonitor",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<Map<String, Object>> getYarnJobMonitor(){
		return applicationMonitoringService.getYarnJobMonitor();
	}
	
	/**
	 * YarnJob hostory (이전)
	 * @param yarnAppMonitorVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYarnJobHistoryBak",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<YarnAppMonitorVO> getYarnJobHistoryBak(YarnAppMonitorVO yarnAppMonitorVO){
		return applicationMonitoringService.getYarnJobHistoryBak(yarnAppMonitorVO);
	}
	
	/**
	 * Metric Registry View 에 저장되어 있는 Metric list 가져오기 
	 * @return
	 */
	@RequestMapping(value="/api/v1/getMetricList",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<Map<String,Object>> getMetricList(){
		return applicationMonitoringService.getMetricList();
	}
	
	/**
	 * 
	 * @param yarnAppMonitorVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYarnJobHistory",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<Map<String,Object>> getYarnJobHistory(YarnAppMonitorVO yarnAppMonitorVO){
		return applicationMonitoringService.getYarnJobHistory(yarnAppMonitorVO);
	}
	
	/**
	 * Node 별 History data 가져옴
	 * @author sehunkim
	 * @param yarnAppMonitorVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYarnJobHistoryPerNode",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<Map<String,Object>> getYarnJobHistoryPerNode(YarnAppMonitorPerNodeVO yarnAppMonitorPerNodeVO){
//		YarnAppMonitorPerNodeVO test = new YarnAppMonitorPerNodeVO();
//		test.setStart_time(20181119180242l);
//		test.setEnd_time(20181120135215l);
//		test.setNode("node01");
		return applicationMonitoringService.getYarnJobHistoryPerNode(yarnAppMonitorPerNodeVO);
//		return applicationMonitoringService.getYarnJobHistoryPerNode(test);
	}
}

package com.xiilab.metric.api;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import com.xiilab.metric.model.MetricRegistryVO;
import com.xiilab.metric.model.YarnAppNodesVO;

@Controller
public class MetricController {
	@Autowired
	private MetricService metricService;
	
	@RequestMapping(value = "/api/v1/metric/test", method = RequestMethod.GET)
	public @ResponseBody List<YarnAppNodesVO> test() {
		return metricService.getYarnAppClientNodes();
	}

	
	/**
	 * check table
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/checkMetricRegistryTable", method = RequestMethod.GET)
	public @ResponseBody Boolean checkMetricRegistryTable() {
		return metricService.checkMetricRegistryTable();
	}
	
	/**
	 * create table and send script files
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/initTable", method = RequestMethod.POST)
	public @ResponseBody Boolean initTable() {
		return metricService.initTable();
	}
	
	/**
	 * get Metric list
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/getMetricList", method = RequestMethod.GET)
	public @ResponseBody List<MetricRegistryVO> getMetricList() {
		return metricService.getMetricList();
	}
	/**
	 * add Metric 
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/addMetric", method = {RequestMethod.POST})
	public @ResponseBody Boolean addMetric(MetricRegistryVO metric) {
		return metricService.addMetric(metric);
	}
	/**
	 * update Metric 
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/updateMetric", method = {RequestMethod.POST})
	public @ResponseBody Boolean updateMetric(MetricRegistryVO metric) {
		return metricService.updateMetric(metric);
	}
	/**
	 * delete metric
	 * @param metric
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/deleteMetric", method = {RequestMethod.POST})
	public @ResponseBody Boolean deleteMetric(MetricRegistryVO metric) {
		return metricService.deleteMetric(metric);
	}
	
	/**
	 * 현재 크론탭 주기 가져오기
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/getMetricCycleTime", method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String,Object> getMetricCycleTime() {
		return metricService.getMetricCycleTime();
	}
	
	
	/**
	 * 연재 메트릭 주기 재설정하기
	 * @param cycle_time
	 * @return
	 */
	@RequestMapping(value = "/api/v1/metric/updateMetricCycleTime", method = {RequestMethod.POST})
	public @ResponseBody Map<String,Object> updateMetricCycleTime(
			@RequestParam(value="cycle_time",required=true) Integer cycle_time
			) {
		return metricService.updateMetricCycleTime(cycle_time);
	}
	
}

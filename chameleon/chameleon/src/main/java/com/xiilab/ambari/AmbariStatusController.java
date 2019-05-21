package com.xiilab.ambari;

import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.web.bind.annotation.RestController;

import com.fasterxml.jackson.annotation.JsonRawValue;

@RestController
public class AmbariStatusController {
	
	@Autowired
	private AmbariStatusServise ambariStatusServise;
	
	/**
	 * ambari node 상태 표시
	 * @return
	 */
	@RequestMapping(value="/api/v1/getHostStatus",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String,String> getHostStatus() {
		return ambariStatusServise.getHostStatus();
	}
	
	
	/**
	 * ambari 저장소 남은용량 표시
	 * @return
	 */
	@RequestMapping(value="/api/v1/getStorageUsage",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String,String> getStorageUsage() {
		return ambariStatusServise.getStorageUsage();
	}
	
	/**
	 * yarn app list 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYarnAppList",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String yarnAppList() {
		return ambariStatusServise.getyarnAllAppList();
	}
	
	/**
	 * yarn stauts 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYarnStatus",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String getYarnStatus() {
		return ambariStatusServise.getYarnStatus();
	}
	
	
	/**
	 * Ambari 노드정보 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/getNodeMatricDataArr",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String getNodeDataArr() {
		return ambariStatusServise.getNodeMatricDataArr();
	}

	
	/**
	 * Ambari 노드정보 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/getNodeMatricDataArrBak",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String getNodeDataArrBak() {
		return ambariStatusServise.getNodeDataArrBak();
	}
	
	/**
	 * 암바리 호스트별로 리스스정보 읽어오기
	 * @param host_name
	 * @return
	 */
	@RequestMapping(value="/api/v1/getNodeMatricDataArrForHost",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String getNodeMatricDataArrForHost(
			@RequestParam(name="host_name",required = true) String host_name
			) {
		return ambariStatusServise.getNodeMatricDataArrForHost(host_name);
	}
	
	
	
	/**
	 * getYARNAppMonitorClientNodes 설치 노드 검색
	 * @return
	 */
	@RequestMapping(value="/api/v1/getYARNAppMonitorClientNodes",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String getYARNAppMonitorClientNodes() {
		return ambariStatusServise.getYARNAppMonitorClientNodes();
	}
	
	/**
	 * Ambari Alert 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/getAlert",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue String getAlert() {
		return ambariStatusServise.getAlert();
	}
	
	
	/**
	 * Ambari LustreManager Metric List
	 * 	 {
	 *		"mds": [
	 *			"mknod",
	 *			"mkdir",
	 *			"open",
	 *			"rename",
	 *			"setattr",
	 *			"rmdir",
	 *			"close",
	 *			"getattr"
	 *		],
	 *		"oss": [
	 *			"Oss2Read",
	 *			"Oss1Write",
	 *			"Oss2Write",
	 *			"Oss1Read"
	 *		]
	 *	}
	 * 
	 * 
	 * @return
	 */
	@RequestMapping(value="/api/v1/getLustreMetrics",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String,Object> getLustreMetrics() {
		return ambariStatusServise.getLustreMetrics();
	}
	
//	@RequestMapping(value="/api/v1/getLustreMetricData",method = {RequestMethod.GET,RequestMethod.POST})
//	public @ResponseBody @JsonRawValue String getLustreMetricData(
//			@RequestParam(name="startTime",required = true) String startTime,
//			@RequestParam(name="endTime",required = true) String endTime
//			) {
//		return ambariStatusServise.getLustreMetricData(startTime,endTime);
//	}
	
	@RequestMapping(value="/api/v1/new_getLustreMetricData",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> getLustreMetricData(
			@RequestParam(name="metricName",required = true) String metricName,
			@RequestParam(name="startTime",required = true) String startTime,
			@RequestParam(name="endTime",required = true) String endTime
			) {
		return ambariStatusServise.new_getLustreMetricData(metricName,startTime,endTime);
	}
	
	
	@RequestMapping(value="/api/v1/getLustreMetricData",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> getLustreMetricData(
			@RequestParam(name="startTime",required = true) String startTime,
			@RequestParam(name="endTime",required = true) String endTime
			) {
		return ambariStatusServise.getLustreMetricData(startTime,endTime);
	}
	
	
	/**
	 * 호스트별로 메트릭 데이터 가져오기
	 * @param metricName
	 * @param startTime
	 * @param endTime
	 * @return
	 */
	@RequestMapping(value="/api/v1/getForHostMetricData",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody @JsonRawValue  String getForHostMetricData(
			@RequestParam(name="metricName",required = true) String metricName,
			@RequestParam(name="startTime",required = true) String startTime,
			@RequestParam(name="endTime",required = true) String endTime
			) {
		return ambariStatusServise.getForHostMetricData(metricName,startTime,endTime);
	}
}

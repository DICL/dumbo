package com.xiilab.ambari;

import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import javax.servlet.ServletContext;

import org.ini4j.Wini;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.support.BasicAuthorizationInterceptor;
import org.springframework.stereotype.Repository;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

@Repository
public class AmbariStatusDAO {

	// ambari agent ini 파일 위
	public final static String AMBARI_AGENT_INI_PATH = "/etc/ambari-agent/conf/ambari-agent.ini";
	// ambari host (테스트용)
	
	
	@Autowired
    private ServletContext servletContext;
	
	// Ambari user name
	@Value("${ambari.username}")
	private String AMBARI_USERNAME;
	// Ambari user password
	@Value("${ambari.password}")
	private String AMBARI_PASSWORD;
	
	@Value("${ambari.ambari_host}")
	private String DEFAULT_AMBARI_SERVER_HOST;
	@Value("${ambari.ambari_port}")
	private String DEFAULT_AMBARI_SERVER_PORT;
	
	
	
	static final Logger LOGGER = LoggerFactory.getLogger(AmbariStatusDAO.class);
	
	
	
	
	
			
	/**
	 * AMBARI Server 호스트상태 가져오기
	 * @return
	 * @throws IOException 
	 */
	public Map<String, String> getHostStatus() {
		
		Map<String, String> result = new HashMap<>();
		
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name;
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl);
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		ObjectMapper mapper = new ObjectMapper();
		try {
			JsonNode actualObj = mapper.readTree(response.getBody());
			result.put("started_count", actualObj.get("Clusters").get("health_report").get("Host/host_state/HEALTHY").asText());
			result.put("total_count", actualObj.get("Clusters").get("total_hosts").asText());
			
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		//result = actualObj.get("items").get(0).get("Clusters").get("cluster_name").asText();
		
		return result;
	}
	
	/**
	 * AMBARI Server 호스트명 가져오기
	 * @return AMBARI 호스트명
	 */
	public String getAmbariServerHostName() {
		
		String result = DEFAULT_AMBARI_SERVER_HOST;
		try {
			// ambari-agent ini file read
			Wini ini = new Wini(new File(AMBARI_AGENT_INI_PATH));
			result = ini.get("server","hostname");
		} catch (Exception e) {
			LOGGER.warn("Not find ambari-agent.ini File default hostname setting : " + DEFAULT_AMBARI_SERVER_HOST + " .....");
			// TODO Auto-generated catch block
			// e.printStackTrace();
		}
		return result;
	}
	
	/**
	 * AMBARI Server 클러스터명 가져오기
	 * @param ambari_host 
	 * @return
	 * @throws IOException 
	 */
	public String getAmbariServerClustreName(String ambari_host) throws IOException {
		String result = null;
		
		RestTemplate restTemplate = new RestTemplate();
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl);
		
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		ObjectMapper mapper = new ObjectMapper();
		JsonNode actualObj = mapper.readTree(response.getBody());
		result = actualObj.get("items").get(0).get("Clusters").get("cluster_name").asText();
		
		return result;
	}
	
	
	
	/**
	 * Yarn Resource Manager 호스트명 가져오기
	 * @param ambari_host
	 * @return
	 * @throws IOException
	 */
	public String getResourceManagerHostName(String ambari_host,String cluster_name) throws IOException {
		String result = null;
		
		RestTemplate restTemplate = new RestTemplate();
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+cluster_name+"/services/YARN/components/RESOURCEMANAGER?fields=host_components";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				.queryParam("fields", "host_components");
		
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		ObjectMapper mapper = new ObjectMapper();
		JsonNode actualObj = mapper.readTree(response.getBody());
		result = actualObj.get("host_components").get(0).get("HostRoles").get("host_name").asText();
		
		return result;
	}
	
	

	/**
	 * AMBARI Server 저장소 정보 가져오기
	 * @return
	 */
	public Map<String, String> getStorageUsage() {
		
		Map<String, String> result = new HashMap<>();
		
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name + "/services/HDFS/components/NAMENODE";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				.queryParam("fields", "ServiceComponentInfo/CapacityUsed,ServiceComponentInfo/CapacityRemaining,ServiceComponentInfo/CapacityTotal");
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		ObjectMapper mapper = new ObjectMapper();
		try {
			JsonNode actualObj = mapper.readTree(response.getBody());
			result.put("CapacityTotal", actualObj.get("ServiceComponentInfo").get("CapacityTotal").asText());
			result.put("CapacityUsed", actualObj.get("ServiceComponentInfo").get("CapacityUsed").asText());
			result.put("CapacityRemaining", actualObj.get("ServiceComponentInfo").get("CapacityRemaining").asText());
		} catch (IOException e) {
			//e.printStackTrace();
			LOGGER.error(e.getMessage());
			return null;
		}
		
		return result;
	}

	
	
	/**
	 * /ws/v1/cluster/apps 가져오기
	 * @return
	 */
	public String yarnAppList(String status_filter) {
		
		RestTemplate restTemplate = new RestTemplate();
		String yarn_resource_manager_host = (String) servletContext.getAttribute("yarn_resource_manager_host");
		String YarnResourceUrl = "http://" + yarn_resource_manager_host + ":" + "8088" + "/ws/v1/cluster/apps";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(YarnResourceUrl);
		
		if(status_filter != null && !"".equals(status_filter)) {
			builder.queryParam("state", status_filter);
		}
		
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}

	/**
	 * /ws/v1/cluster/metrics 가져오기
	 * @return
	 */
	public String getYarnStatus() {
		RestTemplate restTemplate = new RestTemplate();
		String yarn_resource_manager_host = (String) servletContext.getAttribute("yarn_resource_manager_host");
		String YarnResourceUrl = "http://" + yarn_resource_manager_host + ":" + "8088" + "/ws/v1/cluster/metrics";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(YarnResourceUrl);
		
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}

	/**
	 * Ambari 노드정보중에서 LustreFSKernelClient 를 설치한 노드만 출력
	 * @return
	 */
	public String getNodeMatricDataArr() {
	
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name + "/hosts";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				//.queryParam("host_components/HostRoles/component_name", "LustreFSKernelClient")
				//.queryParam("fields", "metrics/cpu,metrics/memory,metrics/disk,metrics/network,host_components/HostRoles/component_name");
				.queryParam("fields", "host_components/HostRoles/component_name");
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}
	
	public String getNodeDataArrBak() {
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name + "/hosts";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				//.queryParam("host_components/HostRoles/component_name", "LustreFSKernelClient")
				.queryParam("fields", "metrics/cpu,metrics/memory,metrics/disk,metrics/network,host_components/HostRoles/component_name");
//				.queryParam("fields", "host_components/HostRoles/component_name");
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}
	
	/**
	 * get Ambari filtering host name
	 * @param host_name
	 * @return
	 */
	public String getNodeMatricDataArrForHost(String host_name) {
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name + "/hosts";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				//.queryParam("host_components/HostRoles/component_name", "LustreFSKernelClient")
				.queryParam("Hosts/host_name", host_name)
				.queryParam("fields", "metrics/cpu,metrics/memory,metrics/disk,metrics/network");
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}

	/**
	 * Alert List 가져오기
	 * @return
	 */
	public String getAlert() {
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name + "/alerts";
		
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				.queryParam("fields", "*")
				.queryParam("!Alert/state", "OK") // 상태가 OK (정상)을 제외한 나머지 
				;
		
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}

	/**
	 * get Ambari LustreManager Metric List
	 * @return
	 */
	public Map<String,Object> getLustreMetrics() {
		
		Map<String,Object> result = new HashMap<>();
		List<String> mds_list = new ArrayList<>();
		List<String> oss_list = new ArrayList<>();
		
		RestTemplate restTemplate = new RestTemplate();
		String metrics_colletor_host = (String) servletContext.getAttribute("metrics_colletor_host");
		int metrics_colletor_port = 6188;
// 		Ambari Lustre Manager Widget 제거로 ambari metric collector 에서 가져옴
		String AmbariResourceUrl = "http://" + metrics_colletor_host + ":" + metrics_colletor_port + "/ws/v1/timeline/metrics/metadata";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				;
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		ObjectMapper mapper = new ObjectMapper();
		try {
			JsonNode actualObj = mapper.readTree(response.getBody());
			JsonNode metric_list = actualObj.get("lustremanager");
			for (int i = 0; metric_list.has(i); i++) {
				String metric_name = metric_list.get(i).get("metricname").asText();
				if(metric_name.contains("Oss")) {
					oss_list.add(metric_name);
				}else if(metric_name.contains("lustre") && !metric_name.contains("Oss") ){
					mds_list.add(metric_name);
				}
			}
			result.put("mds", mds_list);
			result.put("oss", oss_list);
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
			LOGGER.error("404 not found");
			//e.printStackTrace();
		}

// 		Ambari Lustre Manager Widget 제거로 사용불가
		
//		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/stacks/HDP/versions/2.6/services/LUSTREMANAGER/artifacts/metrics_descriptor";
//		UriComponentsBuilder builder = UriComponentsBuilder
//				.fromHttpUrl(AmbariResourceUrl)
//				;
//		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
//		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
//		
//		ObjectMapper mapper = new ObjectMapper();
//		try {
//			JsonNode actualObj = mapper.readTree(response.getBody());
//			
//			Iterator<String> metric_names = actualObj
//				.get("artifact_data")
//				.get("LUSTREMANAGER")
//				.get("HadoopLustreAdapterMgmtService")
//				.get("Component")
//				.get(0)
//				.get("metrics")
//				.get("default")
//				.fieldNames()
//				;
//			
//			while(metric_names.hasNext()) {
//				String key = metric_names.next();
//				if(key.contains("Oss")) {
//					oss_list.add(key);
//				}else {
//					mds_list.add(key);
//				}
//			}
//			
//			result.put("mds", mds_list);
//			result.put("oss", oss_list);
//			
//			
//		} catch (IOException e) {
//			e.printStackTrace();
//		}
		
		return result;
	}
	
	
	/**
	 * 전체 호스트별로 메트릭 데이터 가져오기
	 * @param metricName
	 * @param startTime
	 * @param endTime
	 * @return
	 */
	public String getForHostMetricData(String metricNames, String startTime, String endTime) {
		RestTemplate restTemplate = new RestTemplate();
		String metrics_colletor_host = (String) servletContext.getAttribute("metrics_colletor_host");
		
		int metrics_colletor_port = 6188;
// 		Ambari Lustre Manager Widget 제거로 ambari metric collector 에서 가져옴
		String AmbariResourceUrl = "http://" + metrics_colletor_host + ":" + metrics_colletor_port + "/ws/v1/timeline/metrics";
		
		try {
			UriComponentsBuilder builder = UriComponentsBuilder
					.fromHttpUrl(AmbariResourceUrl)
					.queryParam("metricNames", metricNames)
					.queryParam("appId", "lustremanager")
					.queryParam("precision", "seconds")
					.queryParam("startTime", startTime)
					.queryParam("endTime", endTime)
					.queryParam("hostname", URLEncoder.encode("%", "UTF-8"))
					;
			ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
			ObjectMapper mapper = new ObjectMapper();
			LOGGER.info(builder.toUriString());
			
			return response.getBody();
		} catch (Exception e) {
			e.printStackTrace();
			return null;
		}		
	}
	
	
	/**
	 * 개별적으로 메트릭 데이터 가져오기 ()
	 * @param metricName
	 * @param startTime
	 * @param endTime
	 * @return
	 */
	public Map<String, Object> new_getLustreMetricData(String metricName, String startTime, String endTime) {
		Map<String,Object> result = new HashMap<>();
		RestTemplate restTemplate = new RestTemplate();
		String metrics_colletor_host = (String) servletContext.getAttribute("metrics_colletor_host");
		int metrics_colletor_port = 6188;
// 		Ambari Lustre Manager Widget 제거로 ambari metric collector 에서 가져옴
		String AmbariResourceUrl = "http://" + metrics_colletor_host + ":" + metrics_colletor_port + "/ws/v1/timeline/metrics";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				.queryParam("metricNames", metricName)
				.queryParam("appId", "lustremanager")
				.queryParam("precision", "seconds")
				.queryParam("startTime", startTime)
				.queryParam("endTime", endTime)
				;
		
//		LOGGER.info(builder.toUriString());
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		ObjectMapper mapper = new ObjectMapper();
		try {
			JsonNode actualObj = mapper.readTree(response.getBody());
			JsonNode metrics  = actualObj.get("metrics");
			
			Map<String,Object> tmpMap = new HashMap<>();
			
			for (int i = 0; metrics.has(i); i++) {
				String metric_name = metrics.get(i).get("metricname").asText();
				List<List<Object>> metric_data_list = new ArrayList<>();
				
				
				Iterator<String> timestamp_list = metrics.get(i).get("metrics").fieldNames();
				
				
				while(timestamp_list.hasNext()) {
					List<Object> tmpData = new ArrayList<>();
					
					String timestamp = timestamp_list.next();
					String metric_value = metrics.get(i).get("metrics").get(timestamp).asText();
					tmpData.add(metric_value);
					tmpData.add(timestamp);
					
					metric_data_list.add(tmpData);
				}
				
				tmpMap.put(metric_name, metric_data_list);
				result.put("metrics", tmpMap);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return result;
	}
	
	
	/**
	 * 전체 러스터 메트릭 데이터 가져오기
	 * @param startTime
	 * @param endTime
	 * @return
	 */
	public Map<String,Object> getLustreMetrics(String startTime, String endTime) {
		Map<String,Object> result = new HashMap<>();
		
		Map<String,Object> metric_list = getLustreMetrics();
		List<String> mds_list = (List<String>) metric_list.get("mds");
		List<String> oss_list = (List<String>) metric_list.get("oss");
		
		if(mds_list == null || oss_list == null) {
			String error_message = "not running lustre provider";
			LOGGER.error(error_message);
			result.put("status", false);
			result.put("message", error_message);
			return null;
		}
		
		String fields = "";
		String comma = "";

		for (int i = 0; i < mds_list.toArray().length; i++) {
			String temp = mds_list.get(i);
			if(i > 0) {
				comma = ",";
			}else {
				comma = "";
			}
			fields += comma + temp;
		}
		
		fields += ",";
		
		for (int i = 0; i < oss_list.toArray().length; i++) {
			String temp = oss_list.get(i);
			if(i > 0) {
				comma = ",";
			}else {
				comma = "";
			}
			fields += comma + temp;
		}
		
		
		RestTemplate restTemplate = new RestTemplate();
		String metrics_colletor_host = (String) servletContext.getAttribute("metrics_colletor_host");
		int metrics_colletor_port = 6188;
// 		Ambari Lustre Manager Widget 제거로 ambari metric collector 에서 가져옴
		String AmbariResourceUrl = "http://" + metrics_colletor_host + ":" + metrics_colletor_port + "/ws/v1/timeline/metrics";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				.queryParam("metricNames", fields)
				.queryParam("appId", "lustremanager")
				.queryParam("precision", "seconds")
				.queryParam("startTime", startTime)
				.queryParam("endTime", endTime)
				;
		
//		LOGGER.info(builder.toUriString());
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		ObjectMapper mapper = new ObjectMapper();
		try {
			JsonNode actualObj = mapper.readTree(response.getBody());
			JsonNode metrics  = actualObj.get("metrics");
			
			Map<String,Object> tmpMap = new HashMap<>();
			
			for (int i = 0; metrics.has(i); i++) {
				String metric_name = metrics.get(i).get("metricname").asText();
				List<List<Object>> metric_data_list = new ArrayList<>();
				
				
				Iterator<String> timestamp_list = metrics.get(i).get("metrics").fieldNames();
				
				
				while(timestamp_list.hasNext()) {
					List<Object> tmpData = new ArrayList<>();
					
					String timestamp = timestamp_list.next();
					String metric_value = metrics.get(i).get("metrics").get(timestamp).asText();
					tmpData.add(metric_value);
					tmpData.add(timestamp);
					
					metric_data_list.add(tmpData);
				}
				
				tmpMap.put(metric_name, metric_data_list);
				result.put("metrics", tmpMap);
			}
			
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return result;
	}
	
	
	

	/**
	 * 전체 메트릭 정보 가져오기 (Ambari Lustre Manager Widget 제거로 사용불가)
	 * @param startTime
	 * @param endTime
	 * @return
	 */
//	public String getLustreMetrics(String startTime, String endTime) {
//		
//		RestTemplate restTemplate = new RestTemplate();
//		String ambari_host = (String) servletContext.getAttribute("ambari_host");
//		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
//		
//		Map<String,Object> metric_list = getLustreMetrics();
//		List<String> mds_list = (List<String>) metric_list.get("mds");
//		List<String> oss_list = (List<String>) metric_list.get("oss");
//		
//		String fields = "";
//		String comma = "";
//
//		for (int i = 0; i < mds_list.toArray().length; i++) {
//			String temp = mds_list.get(i);
//			if(i > 0) {
//				comma = ",";
//			}else {
//				comma = "";
//			}
//			fields += comma + temp + "["+ startTime  + "," + endTime +"]";
//		}
//		
//		
//		for (int i = 0; i < oss_list.toArray().length; i++) {
//			String temp = oss_list.get(i);
//			if(i > 0) {
//				comma = ",";
//			}else {
//				comma = "";
//			}
//			fields += comma + temp + "["+ startTime  + "," + endTime +"]";
//		}
//		
//		
//		
//		
//		String AmbariResourceUrl = 
//				"http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT 
//				+ "/api/v1/clusters/"+ambari_cluster_name + "/services/LUSTREMANAGER/components/HadoopLustreAdapterMgmtService?fields="+fields;
////		UriComponentsBuilder builder = UriComponentsBuilder
////				.fromHttpUrl(AmbariResourceUrl)
////				.queryParam("fields", fields)
////				;
//		
//		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
//		ResponseEntity<String> response = restTemplate.exchange(AmbariResourceUrl, HttpMethod.GET, null, String.class);
//		
//		return response.getBody();
//	}

	
	/**
	 * 서버 시작시 AMBARI_METRICS_COLLECTOR 호스트를 찾는 메서드
	 * @param ambari_host
	 * @param cluster_name
	 * @return
	 */
	public String getAmbariMetricCollectorHost(String ambari_host, String cluster_name) {
		String host_name = null;
		RestTemplate restTemplate = new RestTemplate();
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+cluster_name+"/services/AMBARI_METRICS/components/METRICS_COLLECTOR";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				;
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		ObjectMapper mapper = new ObjectMapper();
		
		try {
			JsonNode actualObj = mapper.readTree(response.getBody());
			host_name = actualObj
					.get("host_components")
					.get(0)
					.get("HostRoles")
					.get("host_name")
					.asText()
					;
		}catch (Exception e) {
			host_name = null;
		}
		
		return host_name;
	}

	/**
	 *  getYARNAppMonitorClientNodes 설치 노드 검색
	 * @return
	 */
	public String getYARNAppMonitorClientNodes() {
		RestTemplate restTemplate = new RestTemplate();
		String ambari_host = (String) servletContext.getAttribute("ambari_host");
		String ambari_cluster_name = (String) servletContext.getAttribute("ambari_cluster_name");
		
		String AmbariResourceUrl = "http://" + ambari_host + ":" + DEFAULT_AMBARI_SERVER_PORT + "/api/v1/clusters/"+ambari_cluster_name + "/hosts";
		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
				.queryParam("host_components/HostRoles/component_name", "YARNAppMonitorClient")
				//.queryParam("fields", "host_components/HostRoles/component_name")
				;
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(AMBARI_USERNAME, AMBARI_PASSWORD));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		return response.getBody();
	}

	

}

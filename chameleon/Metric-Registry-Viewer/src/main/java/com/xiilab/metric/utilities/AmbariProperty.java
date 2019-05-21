package com.xiilab.metric.utilities;

import java.util.Map;

import javax.servlet.ServletConfig;

import org.apache.ambari.view.ViewContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.support.BasicAuthorizationInterceptor;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xiilab.metric.model.AmbariViewConfigVO;
import com.xiilab.metric.model.AmbariYarnAppMonitorConfig;

@Component
public class AmbariProperty {
	
	static final Logger LOGGER = LoggerFactory.getLogger(AmbariProperty.class);
	
	@Autowired
	private ServletConfig servletConfig;
	
	
	/**
	 * ambari rest-api 을 통하여 yarnjobmonitor-config-env 설정을 읽어오는 메서드
	 * @return
	 */
	public AmbariYarnAppMonitorConfig getAmbariYarnAppMonitorConfig() {
		AmbariViewConfigVO ambari_view_setting = new AmbariViewConfigVO();
		RestTemplate restTemplate = new RestTemplate();
		
		// 리턴값정의
		AmbariYarnAppMonitorConfig yarnjobmonitor_config_env = new AmbariYarnAppMonitorConfig();
		// json 파셔 호출
		JsonParser jsonParser = new JsonParser();
		
		String ambari_host=""; 
		String ambari_cluster_name ="";
		String ambari_username="";
		String ambari_password="";
		
		// ambari-view 에서 설정한 값을 가져온다
		try {
			ambari_view_setting = readAmbariViewConfigs();
			ambari_host=ambari_view_setting.getAmbari_url();
			ambari_cluster_name =ambari_view_setting.getAmbari_containername();
			ambari_username=ambari_view_setting.getAmbari_username();
			ambari_password=ambari_view_setting.getAmbari_password();
		// 만약 ambari-view 정보를 읽어올수 없는 경우에는 사전에 설정된 값으로 가져온다
		}catch (Exception e) {
			LOGGER.warn(e.getMessage());
			ambari_host="http://192.168.0.23:8080"; 
			ambari_cluster_name ="supercom_test";
			ambari_username="admin";
			ambari_password="admin";
		}
	
		
		
		try {
			// 192.168.1.191:8080/api/v1/clusters/supercom_test/?fields=Clusters/desired_configs 호출
			String AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "/";
			UriComponentsBuilder builder = UriComponentsBuilder
					.fromHttpUrl(AmbariResourceUrl)
					.queryParam("fields", "Clusters/desired_configs")
					;
			// basic auth 설정
			restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(ambari_username, ambari_password));
			ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
			
			// 출력된 결과값을 json object 로 파싱
			JsonObject ambari_json_info = (JsonObject) jsonParser.parse(response.getBody());
			// json 내부에서 yarnjobmonitor-config-env tag 명 가져오기 (현재 적용된 환경설정값)
			String yarnjobmonitor_config_env_tag_name = ambari_json_info
					.get("Clusters")
					.getAsJsonObject()
					.get("desired_configs")
					.getAsJsonObject()
					.get("yarnjobmonitor-config-env")
					.getAsJsonObject()
					.get("tag")
					.getAsString();
			
			// 192.168.1.191:8080/api/v1/clusters/supercom_test/configurations 호출
			AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "/configurations";
			builder = UriComponentsBuilder
					.fromHttpUrl(AmbariResourceUrl)
					.queryParam("type", "yarnjobmonitor-config-env")
					// 이전에 가져온 tag 값 삽입
					.queryParam("tag", yarnjobmonitor_config_env_tag_name)
					;
			// basic auth
			restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(ambari_username, ambari_password));
			response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
			
			String response_body = response.getBody();
			// 결과값을 로그로 출력
			LOGGER.info(response_body);
			// 결과 값을 json 으로 파싱
			ambari_json_info = (JsonObject) jsonParser.parse(response_body);
			// json 내부에서 properties 값 가져오기
			JsonObject yarnjobmonitor_config_env_properties = ambari_json_info
					.get("items")
					.getAsJsonArray()
					.get(0)
					.getAsJsonObject()
					.get("properties")
					.getAsJsonObject();
			
			// properties 에서 일치하는 값과 매칭하여 삽입후 반환
			yarnjobmonitor_config_env.setTimescaleDB_database(yarnjobmonitor_config_env_properties.get("TimescaleDB_database").getAsString());
			yarnjobmonitor_config_env.setTimescaleDB_url(yarnjobmonitor_config_env_properties.get("TimescaleDB_ip").getAsString());
			yarnjobmonitor_config_env.setTimescaleDB_port(yarnjobmonitor_config_env_properties.get("TimescaleDB_port").getAsString());
			yarnjobmonitor_config_env.setTimescaleDB_password(yarnjobmonitor_config_env_properties.get("TimescaleDB_password").getAsString());
			yarnjobmonitor_config_env.setTimescaleDB_username(yarnjobmonitor_config_env_properties.get("TimescaleDB_username").getAsString());
			
			
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
		}
		
		
		return yarnjobmonitor_config_env;
	}
	
	
	/**
	 * Ambari View Setting 에서 설정한 환경설정정보를 읽어온다
	 * @return
	 */
	public Map<String,String> getServerConfigs() {
		Map<String,String> result = null;
		//ViewContext viewContext = (ViewContext) request.getSession().getServletContext().getAttribute(ViewContext.CONTEXT_ATTRIBUTE);
		// ambari-view 에서 property 값을 읽어 올려면 ServletConfig 에서 읽어 와야함
		
		
		
		
		
		try {
			ViewContext viewContext = (ViewContext) servletConfig.getServletContext().getAttribute(ViewContext.CONTEXT_ATTRIBUTE);
			if(viewContext != null) {
				result = viewContext.getProperties();
			}
		} catch (Exception e) {
			//e.printStackTrace();
		}
		
		return result;
	}
	
	
	/**
	 * server property 을 읽어와서 AmbariViewConfigVO 적재하는 메서드
	 * @return
	 */
	public AmbariViewConfigVO readAmbariViewConfigs() {
		AmbariViewConfigVO result = new AmbariViewConfigVO();
		Map<String,String> tmp = getServerConfigs();
		
		result.setAmbari_containername(tmp.get("ambari.metric_server.containername"));
		result.setAmbari_url(tmp.get("ambari.metric_server.url"));
		result.setAmbari_username(tmp.get("ambari.metric_server.username"));
		result.setAmbari_password(tmp.get("ambari.metric_server.password"));
		
		return result;
	}
	
	
}

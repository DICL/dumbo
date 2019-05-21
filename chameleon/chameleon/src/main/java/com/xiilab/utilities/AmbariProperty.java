package com.xiilab.utilities;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.support.BasicAuthorizationInterceptor;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xiilab.models.AmbariYarnAppMonitorConfig;


@Component
public class AmbariProperty {
	static final Logger LOGGER = LoggerFactory.getLogger(AmbariProperty.class);
	
	// Ambari user name
	@Value("${ambari.username}")
	private String AMBARI_USERNAME;
	// Ambari user password
	@Value("${ambari.password}")
	private String AMBARI_PASSWORD;
		
	@Value("${ambari.ambari_host}")
	private String DEFAULT_AMBARI_SERVER_HOST;
	@Value("${ambari.ambari_cluster_name}")
	private String DEFAULT_AMBARI_CLUSTER_NAME;
	@Value("${ambari.ambari_port}")
	private String DEFAULT_AMBARI_SERVER_PORT;
	
	/**
	 * ambari rest-api 을 통하여 yarnjobmonitor-config-env 설정을 읽어오는 메서드
	 * @return
	 */
	public AmbariYarnAppMonitorConfig getAmbariYarnAppMonitorConfig() throws Exception{
		RestTemplate restTemplate = new RestTemplate();
		
		// 리턴값정의
		AmbariYarnAppMonitorConfig yarnjobmonitor_config_env = new AmbariYarnAppMonitorConfig();
		// json 파셔 호출
		JsonParser jsonParser = new JsonParser();
		
		String ambari_host="http://"+DEFAULT_AMBARI_SERVER_HOST+":"+DEFAULT_AMBARI_SERVER_PORT; ; 
		String ambari_cluster_name=DEFAULT_AMBARI_CLUSTER_NAME;
		String ambari_username=AMBARI_USERNAME;
		String ambari_password=AMBARI_PASSWORD;
		
		
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
			
			
		
		
		
		return yarnjobmonitor_config_env;
	}


}

package com.xiilab.metric.api;

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;

import javax.annotation.Resource;

import org.mybatis.spring.SqlSessionTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.support.BasicAuthorizationInterceptor;
import org.springframework.stereotype.Repository;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xiilab.metric.model.AmbariViewConfigVO;
import com.xiilab.metric.model.MetricRegistryVO;
import com.xiilab.metric.model.YarnAppNodesVO;
import com.xiilab.metric.utilities.AmbariProperty;

@Repository
public class MetricDAO {
	
	static final Logger LOGGER = LoggerFactory.getLogger(MetricDAO.class);
	public final static String AMBARI_AGENT_INI_PATH = "/etc/ambari-agent/conf/ambari-agent.ini";
	
	@Autowired
	private AmbariProperty ambariProperty;
	
	@Autowired
	@Resource(name="ambari-sqlSession")
	private SqlSessionTemplate sqlSession;
	
	@Resource(name="timescale-sqlSession")
	private SqlSessionTemplate timescale_sqlSession;
	
//	@Autowired
//	private AmbariProperty ambariProperty;
	
	// true : 하드코딩된 정보 가져오기 false : ambari-view 에서 설정된 값들을 가져오기
	public final static boolean is_develop = false;
	
	/**
	 * ambari 환경설정 가져오기
	 * @return
	 */
	private AmbariViewConfigVO getAmbariViewConfig() {
		
		String ambari_host="";
		String ambari_cluster_name ="";
		String ambari_username="";
		String ambari_password="";
		
		AmbariViewConfigVO ambari_view_setting = new AmbariViewConfigVO();
		
		//나중에 ambari-view 에서 가져올것
		if(is_develop) {
			ambari_host="http://192.168.0.13:8080"; 
			ambari_cluster_name ="supercom_test";
			ambari_username="admin";
			ambari_password="admin";
			
			ambari_view_setting.setAmbari_url(ambari_host);
			ambari_view_setting.setAmbari_containername(ambari_cluster_name);
			ambari_view_setting.setAmbari_password(ambari_password);
			ambari_view_setting.setAmbari_username(ambari_username);
			return ambari_view_setting;
		}else {
			try {
				ambari_view_setting = ambariProperty.readAmbariViewConfigs();
				LOGGER.warn("conatiner name : {}",ambari_view_setting.getAmbari_containername());
				LOGGER.warn("ambari url : {}",ambari_view_setting.getAmbari_url());
				LOGGER.warn("ambari user name : {}",ambari_view_setting.getAmbari_username());
				LOGGER.warn("ambari pssword : {}",ambari_view_setting.getAmbari_password());
				return ambari_view_setting;
			}catch (Exception e) {
				LOGGER.warn("error statut 1 :{}",e.getMessage());
				ambari_host="http://192.168.0.23:8080"; 
				ambari_cluster_name ="supercom_test";
				ambari_username="admin";
				ambari_password="admin";
				
				ambari_view_setting.setAmbari_url(ambari_host);
				ambari_view_setting.setAmbari_containername(ambari_cluster_name);
				ambari_view_setting.setAmbari_password(ambari_password);
				ambari_view_setting.setAmbari_username(ambari_username);
				return ambari_view_setting;
			}
		}
	}
	
	/**
	 * 
	 * @return
	 */
	public String getYarnAppMonitorClientFilePath() {
		String result = null;
		
		
		RestTemplate restTemplate = new RestTemplate();
		// json 파셔 호출
		JsonParser jsonParser = new JsonParser();
		
		String AmbariResourceUrl = "";
		String ambari_host="";
		String ambari_cluster_name ="";
		String ambari_username="";
		String ambari_password="";
		
		AmbariViewConfigVO ambariViewConfigVO = getAmbariViewConfig();
		ambari_host = ambariViewConfigVO.getAmbari_url();
		ambari_cluster_name = ambariViewConfigVO.getAmbari_containername();
		ambari_username = ambariViewConfigVO.getAmbari_username();
		ambari_password = ambariViewConfigVO.getAmbari_password();
		
		// json object 파싱
		try {
			AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "";
			UriComponentsBuilder builder = UriComponentsBuilder
					.fromHttpUrl(AmbariResourceUrl)
					.queryParam("fields", "Clusters/desired_configs")
					;
			restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(ambari_username, ambari_password));
			ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
			
			
			JsonObject ambari_configs = (JsonObject) jsonParser.parse(response.getBody());
			
			// 태그 이름 추출
			String tagname = ambari_configs.get("Clusters")
				.getAsJsonObject().get("desired_configs")
				.getAsJsonObject().get("yarnjobmonitor-config-env")
				.getAsJsonObject().get("tag").getAsString();
			
			// 태그명을 토대로 ambari config 내용 불려오기
			AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "/configurations";
			builder = UriComponentsBuilder
						.fromHttpUrl(AmbariResourceUrl)
						.queryParam("type", "yarnjobmonitor-config-env")
						.queryParam("tag", tagname)
					;
			restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(ambari_username, ambari_password));
			response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
			JsonObject lustrefs_config_env = (JsonObject) jsonParser.parse(response.getBody());
			
			// 환경설정 내용 가져오기
			result = lustrefs_config_env
					.get("items")
					.getAsJsonArray().get(0)
					.getAsJsonObject().get("properties")
					.getAsJsonObject().get("location")
					.getAsString();
			
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
			result = null;
		}
		
		
		return result;
	}
	
	/**
	 * getYarnAppClient 가 설치된 호스트 추출
	 * @return
	 */
	public List<YarnAppNodesVO> getYarnAppClientNodes(){
		List<YarnAppNodesVO> result = new ArrayList<>();
		RestTemplate restTemplate = new RestTemplate();
		
		
		// json 파셔 호출
		JsonParser jsonParser = new JsonParser();
		
		String AmbariResourceUrl = "";
		String ambari_host="";
		String ambari_cluster_name ="";
		String ambari_username="";
		String ambari_password="";
		
		AmbariViewConfigVO ambariViewConfigVO = getAmbariViewConfig();
		ambari_host = ambariViewConfigVO.getAmbari_url();
		ambari_cluster_name = ambariViewConfigVO.getAmbari_containername();
		ambari_username = ambariViewConfigVO.getAmbari_username();
		ambari_password = ambariViewConfigVO.getAmbari_password();
		try {
			AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "/hosts";
			UriComponentsBuilder builder = UriComponentsBuilder
					.fromHttpUrl(AmbariResourceUrl)
					.queryParam("fields", "host_components/HostRoles/component_name")
					.queryParam("host_components/HostRoles/component_name", "YARNAppMonitorClient")
					;
			restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(ambari_username, ambari_password));
			ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
			
			
			JsonObject ambari_json_info = (JsonObject) jsonParser.parse(response.getBody());
			// items 라는 오브젝트 추출
			JsonArray ambari_list = (JsonArray) ambari_json_info.get("items");
			for (int i = 0; i < ambari_list.size(); i++) {
				YarnAppNodesVO tmp_host_info = new YarnAppNodesVO();
				JsonObject host_information = (JsonObject) ambari_list.get(i);
				// 해당 호스트내 Hosts 추출
				String host_name = host_information.get("Hosts").getAsJsonObject().get("host_name").getAsString();
				LOGGER.info("host name : {}",host_name);
				tmp_host_info.setHost_name(host_name);
				tmp_host_info.setUser_id("root");
				tmp_host_info.setSsh_port(22);
				tmp_host_info.setPrivate_key("~/.ssh/id_rsa");
				result.add(tmp_host_info);
			}
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
			result = null;
		}
		
		
		
		return result;
	}
	
	
	/**
	 * 테스트
	 * @return
	 */
	public String test_timescale() {
		return timescale_sqlSession.selectOne("com.xiilab.timescale.api.test");
	}
	

	/**
	 * name, col_name 만 검색 (데이블이 없을경우 테이블 생성로직도 추가)
	 * @return
	 */
	public List<MetricRegistryVO> getMetricList() {
		// 만약 테이블이 없으면 테이블을 생성
		if(checkMetricRegistryTable().size() <= 0) {
			create_metric_registry_table();
		}
		
		if(checkMetricCycleTable().size() <= 0) {
			create_metric_cycle_table();
		}
		// 이름과 컬럼만 리스트
		return sqlSession.selectList("com.xiilab.metric.api.get_metric_list");
	}
	
	
	/**
	 * 처음 metric registry 시작시 동작되는 메서드
	 * @return
	 */
	public Boolean create_metric_cycle_table() {
		boolean result = false;
		result = sqlSession.update("com.xiilab.metric.api.create_metric_cycle_table") > 0;
		result = sqlSession.insert("com.xiilab.metric.api.insert_metric_cycle_default_data") > 0;
		return result;
	}
	
	/**
	 * 처음 metric registry 시작시 동작되는 메서드
	 * @return
	 */
	public Boolean create_metric_registry_table() {
		boolean result = false;
		// 테이블 생성
		result = sqlSession.update("com.xiilab.metric.api.create_metric_registry_table") > 0;
		// 생성된 테이블로 디폴트 데이터 생성
		result = sqlSession.update("com.xiilab.metric.api.insert_metric_registry_default_data") > 0;

		
		
		
		

//		// 테이블 생성용 시퀀스 생성
//		result = check_sequence();
//		// 테이블 네임 생성
//		// 현재 날짜를 삽입
//		TimeZone tz;
//		tz = TimeZone.getTimeZone("Asia/Seoul"); 
//		DateFormat df = new SimpleDateFormat("yyyyMMddHHmmss");
//		df.setTimeZone(tz);
//		String time_label = df.format(new Date());
//		// 테이블 네임 생성
//		String table_name = "apptable_1_"+time_label;
		return result;
	}

	/**
	 * 전체 내용을 출력
	 * @return
	 */
	public List<MetricRegistryVO> get_all_metric_list() {
		return sqlSession.selectList("com.xiilab.metric.api.get_all_metric_list");
	}

	/**
	 * 암바리 postgreSQL 에서 테이블 검색
	 * @return
	 */
	public List<String> checkMetricRegistryTable() {
		return sqlSession.selectList("com.xiilab.metric.api.check_metric_registry_table");
	}

	public List<String> checkMetricCycleTable() {
		return sqlSession.selectList("com.xiilab.metric.api.check_metric_cycle_table");
	}
	/**
	 * 메트릭 추가
	 * @param metric
	 * @return
	 */
	public Boolean addMetric(MetricRegistryVO metric) {
		return sqlSession.insert("com.xiilab.metric.api.addMetric",metric) > 0;
	}

	/**
	 * 메트릭 내용보기
	 * @param metric
	 * @return
	 */
	public MetricRegistryVO viewMetric(MetricRegistryVO metric) {
		return sqlSession.selectOne("com.xiilab.metric.api.viewMetric",metric);
	}

	/**
	 * 메트릭 업데이트
	 * @param metric
	 * @return
	 */
	public Boolean updateMetric(MetricRegistryVO metric) {
		return sqlSession.insert("com.xiilab.metric.api.updateMetric",metric) > 0;
	}

	/**
	 * 메트릭 삭제
	 * @param metric
	 * @return
	 */
	public Boolean deleteMetric(MetricRegistryVO metric) {
		return sqlSession.delete("com.xiilab.metric.api.deleteMetric",metric)> 0;
	}
	
	/**
	 * 하이퍼 테이블 리스트 가져오시
	 * @param table_name
	 * @return
	 */
	public List<String> find_hyper_table(String table_name) {
		return timescale_sqlSession.selectList("com.xiilab.timescale.api.find_timescale_table",table_name);
	}


	/**
	 * timescaleDB 에 테이블 생성 및 하이퍼테이블 로 변경
	 * @param table_info
	 * @return
	 */
	public Boolean create_timescale(Map<String, Object> table_info) {
		LOGGER.info("table info {}",table_info.getClass());
//		List<MetricRegistryVO> colum_list = (List<MetricRegistryVO>) table_info.get("colum_list");
//		
//		String sql = 
//		"CREATE TABLE "+ (table_info.get("table_name")) +"\n" + 
//		"    	(\n" + 
//		"    		create_date timestamp with time zone NOT NULL\n" + 
//		"    		,pid character varying(255)\n" + 
//		"    		,application_id character varying(255)\n" + 
//		"    		,container_id character varying(255)\n" + 
//		"    		,node character varying(255)\n" ;
//		
//		for (MetricRegistryVO metric : colum_list) {
//			sql += "    		," + metric.getName() + " character varying(255)\n";
//		}
//		
//		
//		sql += "    	);";
		
		
		return timescale_sqlSession.insert("com.xiilab.timescale.api.create_timescale",table_info) > 0;
	}
	
	/**
	 * 기존에 있는 테이블을 하이퍼 테이블로 전환 (미사용)
	 * @param table_info
	 * @return
	 */
	public Boolean create_hypertable(Map<String, Object> table_info) {
		try {
			timescale_sqlSession.selectOne("com.xiilab.timescale.api.create_hypertable",table_info);
			return true;
		} catch (Exception e) {
			LOGGER.error(e.getMessage());
			return false;
		}
	}
	
	/**
	 * hyper_table 생성용 시퀀스 만들기
	 * @return
	 */
	public Boolean check_sequence() {
		int last_sequence = 0;
		try {
			last_sequence = (Integer) timescale_sqlSession.selectOne("com.xiilab.timescale.api.check_timescale_sequence");
		} catch (Exception e) {
			last_sequence = 0;
		}
		if( last_sequence <= 0  ) {
			LOGGER.info("create sequence : apptable_seq");
			return create_timescale_sequence();
		}else {
			return true;
		}
	}
	
	public Boolean create_timescale_sequence() {
		return timescale_sqlSession.update("com.xiilab.timescale.api.create_timescale_sequence") > 0;
	}
	
	
	/**
	 * hyper_table 시퀀스 다음 숫자 가져오기
	 * @return
	 */
	public Long get_sequence_next_val() {
		return timescale_sqlSession.selectOne("com.xiilab.timescale.api.get_sequence_next_val");
	}

	/**
	 * 현재 크론탭 주기 가져오기
	 * @return
	 */
	public Map<String, Object> getMetricCycleTime() {
		Map<String, Object> result = new HashMap<>();
		Integer cycle_time = sqlSession.selectOne("com.xiilab.metric.api.getMetricCycleTime");
		result.put("status", cycle_time != null);
		result.put("cycle_time", cycle_time);
		return result;
	}

	public int updateMetricCycleTime(Integer cycle_time) {
		return sqlSession.update("com.xiilab.metric.api.updateMetricCycleTime",cycle_time);
	}

}

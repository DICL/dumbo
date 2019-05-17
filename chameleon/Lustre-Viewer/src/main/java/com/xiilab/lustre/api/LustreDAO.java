package com.xiilab.lustre.api;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;

import org.ini4j.Wini;
import org.mybatis.spring.SqlSessionTemplate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpMethod;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.support.BasicAuthorizationInterceptor;
import org.springframework.stereotype.Repository;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;
import org.springframework.web.util.UriComponentsBuilder;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.xiilab.lustre.model.AmbariViewConfigVO;
import com.xiilab.lustre.model.DiskInforVO;
import com.xiilab.lustre.model.LustreFileSystemListVO;
import com.xiilab.lustre.model.LustreLogVO;
import com.xiilab.lustre.model.LustreNodesVO;
import com.xiilab.lustre.utilities.AmbariProperty;
import com.xiilab.lustre.utilities.SshClient;

@Repository
public class LustreDAO {
	
	@Autowired
	private SqlSessionTemplate sqlSession;
	
	@Autowired
	private AmbariProperty ambariProperty;
	
	public final static String AMBARI_AGENT_INI_PATH = "/etc/ambari-agent/conf/ambari-agent.ini";
	static final Logger LOGGER = LoggerFactory.getLogger(LustreDAO.class);
	
	// ambari lustre service
	// true : 하드코딩된 정보 가져오기 false : ambari-view 에서 설정된 값들을 가져오기
	public final static boolean is_develop = false;
	
	
	public final static String LUSTRE_MDS_COMPONENTS = "LustreMDSMgmtService";
	public final static String LUSTRE_OSS_COMPONENTS = "LustreOSSMgmtService";
	public final static String LUSTRE_CLIENT_COMPONENTS = "LustreClient";
	
	@Autowired
	private SshClient sshClient;
	
	
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
			ambari_host="http://192.168.1.191:8080"; 
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
				return ambari_view_setting;
			}catch (Exception e) {
				LOGGER.warn(e.getMessage());
				ambari_host="http://192.168.1.191:8080"; 
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
	 * 호스트정보 가져오기
	 * @return
	 */
	public String getNodeMatricDataArr() {
		
		RestTemplate restTemplate = new RestTemplate();
		
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
		
		
		AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "/hosts";
		

		UriComponentsBuilder builder = UriComponentsBuilder
				.fromHttpUrl(AmbariResourceUrl)
//				.queryParam("host_components/HostRoles/component_name", "LustreMDSMgmtService")
				.queryParam("fields", "host_components/HostRoles/component_name")
//				.queryParam("fields", "metrics/cpu,metrics/memory,metrics/disk,metrics/network,host_components/HostRoles/component_name")
				;
		restTemplate.getInterceptors().add(new BasicAuthorizationInterceptor(ambari_username, ambari_password));
		ResponseEntity<String> response = restTemplate.exchange(builder.toUriString(), HttpMethod.GET, null, String.class);
		
		
		
		return response.getBody();
	}
	
	
	
	// ambari LUSTREMANAGER Service Config 내용 가져오기
	// - 미사용
	public String getMdtFsname() {
		
		RestTemplate restTemplate = new RestTemplate();
		
		String result;
		
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
				.getAsJsonObject().get("lustrefs-config-env")
				.getAsJsonObject().get("tag").getAsString();
			
			// 태그명을 토대로 ambari config 내용 불려오기
			AmbariResourceUrl = ambari_host + "/api/v1/clusters/"+ ambari_cluster_name + "/configurations";
			builder = UriComponentsBuilder
					.fromHttpUrl(AmbariResourceUrl)
					.queryParam("type", "lustrefs-config-env")
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
				.getAsJsonObject().get("mdt_fsname")
				.getAsString();
			
			
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
			result = null;
		}
		
		
		return result;
	}
	
	
	/**
	 * ambari api로 호출된 내용을 토대로 러스터 서비스가 설치된 노드를 탐색하는 메서드
	 * @return
	 */
	public List<LustreNodesVO> findLustreNodesForAmbari() {
		
		List<LustreNodesVO> result = new ArrayList<>();
		LustreNodesVO temp_lustre_info = null;
		
		int mds_index = 0;
		int oss_index = 0;
		
		// json 파셔 호출
		JsonParser jsonParser = new JsonParser();
		// ambari component 정보 추출
		String ambari_components_str = getNodeMatricDataArr();
		
		try {
			// json object 파싱
			JsonObject ambari_json_info = (JsonObject) jsonParser.parse(ambari_components_str);
			// items 라는 오브젝트 추출
			JsonArray ambari_list = (JsonArray) ambari_json_info.get("items");
			
			for (int i = 0; i < ambari_list.size(); i++) {
				JsonObject host_information = (JsonObject) ambari_list.get(i);
				// 해당 호스트내 components 추출
				JsonArray conponent_list = (JsonArray) host_information.get("host_components");
				
				
				String host_name = host_information.get("Hosts").getAsJsonObject().get("host_name").getAsString();
				temp_lustre_info = null;
				
				for (int j = 0; j < conponent_list.size(); j++) {
					JsonObject component_information = (JsonObject) conponent_list.get(j);
					String component_name = component_information.get("HostRoles").getAsJsonObject().get("component_name").getAsString();
					if(LUSTRE_CLIENT_COMPONENTS.equals(component_name)) {
						temp_lustre_info = new LustreNodesVO();
						temp_lustre_info.setHost_name(host_name);
						temp_lustre_info.setIndex(0);
						temp_lustre_info.setNode_type("CLIENT");
					}
				}
				for (int j = 0; j < conponent_list.size(); j++) {
					JsonObject component_information = (JsonObject) conponent_list.get(j);
					String component_name = component_information.get("HostRoles").getAsJsonObject().get("component_name").getAsString();
					if(LUSTRE_OSS_COMPONENTS.equals(component_name)) {
						temp_lustre_info = new LustreNodesVO();
						temp_lustre_info.setHost_name(host_name);
						temp_lustre_info.setIndex(oss_index);
						temp_lustre_info.setNode_type("OSS");
						oss_index ++;
					}
				}
				for (int j = 0; j < conponent_list.size(); j++) {
					JsonObject component_information = (JsonObject) conponent_list.get(j);
					String component_name = component_information.get("HostRoles").getAsJsonObject().get("component_name").getAsString();
					if(LUSTRE_MDS_COMPONENTS.equals(component_name)) {
						temp_lustre_info = new LustreNodesVO();
						temp_lustre_info.setHost_name(host_name);
						temp_lustre_info.setIndex(mds_index);
						temp_lustre_info.setNode_type("MDS");
						mds_index ++;
					}
				}
				
				if(temp_lustre_info != null) {
					result.add(temp_lustre_info);
				}
				
				
				
				
				
//				for (int j = 0; j < conponent_list.size(); j++) {
//					JsonObject component_information = (JsonObject) conponent_list.get(j);
//					String component_name = component_information.get("HostRoles").getAsJsonObject().get("component_name").getAsString();
//					String host_name = component_information.get("HostRoles").getAsJsonObject().get("host_name").getAsString();
//					if(LUSTRE_MDS_COMPONENTS.equals(component_name)) {
//						temp_lustre_info = new LustreNodesVO();
//						temp_lustre_info.setHost_name(host_name);
//						temp_lustre_info.setIndex(mds_index);
//						temp_lustre_info.setNode_type("MDS");
//						result.add(temp_lustre_info);
//						mds_index ++;
//					}else if(LUSTRE_OSS_COMPONENTS.equals(component_name)) {
//						temp_lustre_info = new LustreNodesVO();
//						temp_lustre_info.setHost_name(host_name);
//						temp_lustre_info.setIndex(oss_index);
//						temp_lustre_info.setNode_type("OSS");
//						result.add(temp_lustre_info);
//						oss_index ++;
//					}else if(LUSTRE_CLIENT_COMPONENTS.equals(component_name)) {
//						temp_lustre_info = new LustreNodesVO();
//						temp_lustre_info.setHost_name(host_name);
//						temp_lustre_info.setIndex(0);
//						temp_lustre_info.setNode_type("CLIENT");
//						result.add(temp_lustre_info);
//					}
//				}
			
			}
			
		} catch (Exception e) {
			//e.printStackTrace();
			LOGGER.error(e.getMessage());
			return result;
		}
		
		
		return result;
	}
	


	/**
	 * 181227 je.kim
	 * 암바리 API을 통하여 러스터 정보들을 삽입하는 메서드
	 * @param file_system
	 * @return
	 */
	public Boolean createLustreNodeForAabari(LustreFileSystemListVO file_system) {
		
		// 암바리 API을 통하여 데이터 읽어오기
		List<LustreNodesVO> ambari_list = findLustreNodesForAmbari();
//		// 파일시스텀정보들을 읽어오기
		LustreFileSystemListVO file_system_info = viewFileSystem(file_system);
//		LustreFileSystemListVO file_system_info = file_system;
		
		// 널값일경우 실패
//		if(file_system_info == null || file_system_info.getNum() == null) {
//			return false;
//		}
//		
//		// 디비에 파일시스템정보가 없을경우 실패
		if(file_system_info == null) {
			return false;
		}
		
		
		// 새로 갱신할 리스트 정보
		List<LustreNodesVO> result_list = new ArrayList<>();
		// 디비에 삽입할 정보
		LustreNodesVO insert_nodes = new LustreNodesVO();
		for (LustreNodesVO temp_lustre : ambari_list) {
			// 읽어온 파일시스템 기본키을 삽입
			temp_lustre.setFile_system_num(file_system_info.getNum());
			result_list.add(temp_lustre);
		}
		
		// 암바리에서 설치정보가 없을경우 실패
		if(ambari_list.size()>0) {
			insert_nodes.setList(result_list);
			return sqlSession.insert("com.xiilab.lustre.api.insert_lustre_list",insert_nodes) > 0;
		}else {
			return false;
		}
		
	}
	
	
	
	
	/**
	 * 디비에 적재되어 있는 러스터정보을 읽어오고 없으면 암바리를 통하여 동기화 하는 메서드
	 * @param file_system 
	 * @return
	 */
	public boolean syncLustreTable(LustreFileSystemListVO file_system) {
		
		LustreNodesVO insert_lustre_info = new LustreNodesVO();
		
		// 검색할 러스터 노드
		LustreNodesVO search_node =  new LustreNodesVO();
		search_node.setFile_system_num(file_system.getNum());
		List<LustreNodesVO> db_list = getLustreNodes(search_node); //파일 시스템 번호로 탐색
		
		List<LustreNodesVO> ambari_list = findLustreNodesForAmbari();
		
		file_system = viewFileSystem(file_system);
		
		
		// 디비에 저장되어 있지 않는경우 암바리을 통하여 디비에 데이터 적재
		if(db_list.size() == 0) {
			LOGGER.info("None Create Lustre Node Data Sync Ambari Api..");
			
			
			insert_lustre_info.setList(findLustreNodesForAmbari()); 
		
		// 디비에 적재 되었이지만 암바리에서 호출된 러스터노드 갯수와 디비에 적재되어 있는 러스터 노드가 일치하지 않는경우 동기화
		}else if (db_list.size() > 0&& db_list.size() < ambari_list.size() ) {
			
			LOGGER.info("new Install Ambari Service Data Base Nodes =>{}, Ambari Lustre Manager Node =>{}",db_list.size(),ambari_list.size());
			List<LustreNodesVO> insert_list = new ArrayList<>();
			
			List<String> remove_list =  new ArrayList<>();
			
			// ambari 에서 읽어온 리스트와 디비에 적재된 리스트 비교하여 동일한 호스트 네임 적재
			for (LustreNodesVO node_info : db_list) {
				for (LustreNodesVO ambari_node : ambari_list) {
					if(node_info.getHost_name().equals(ambari_node.getHost_name())) {
						remove_list.add(ambari_node.getHost_name());
					}
				}
			}
			
			// 키값찾기
			int oss_index = getNodeLastIndex("OSS",file_system.getNum());
			int mds_index = getNodeLastIndex("MDS",file_system.getNum());
			
			// ambari 에서 읽어온 리스트에서 일치하지 않는 리스트만 적재후
			for (LustreNodesVO ambari_node : ambari_list) {
				
//				// KISTI 에서 Add file system 을 제외하고 OSS 네트워크 디바이스를 수정하지 않는다고 하여
//				// 이전에 추가된 내용을 가지고 네트워크 디바이스명을 가져오기로함
//				LustreNodesVO mds_sample_node = null; // 참고할 MDS
//				LustreNodesVO oss_sample_node = null; // 참고할 OSS
//				
//				switch (ambari_node.getNode_type()) {
//				case "MDS": // 탐색한 노드가 MDS 일경우
//					if(ambari_node.getNetwork_device() != null) {
//						mds_sample_node = ambari_node; // 참고할 노드정보를 갱신
//					}
//					break;
//				case "OSS": // 탐색한 노드가 OSS 일경우
//					if(ambari_node.getNetwork_device() != null) {
//						oss_sample_node = ambari_node; // 참고할 노드정보를 갱신
//					}
//					break;
//				default:
//					break;
//				}
				
				
				// 암바리에서 가져온 노드와 디비에서 가져온 노드정보가 일치 하지 않을때
				if(!remove_list.contains(ambari_node.getHost_name())) {
					// je.kim file system key 키값 삽입
					ambari_node.setFile_system_num(file_system.getNum());
					// 마직막 인덱스 값을 삽입후에
					if(ambari_node.getNode_type().equals("MDS")) {
						mds_index ++;
						ambari_node.setIndex(mds_index);
						// 이전에 참고한 네트워크 디바이스 명으로 교체
						//ambari_node.setNetwork_device(mds_sample_node.getNetwork_device());
					}else if(ambari_node.getNode_type().equals("OSS")) {
						oss_index ++;
						ambari_node.setIndex(oss_index);
						// 이전에 참고한 네트워크 디바이스 명으로 교체
						//ambari_node.setNetwork_device(oss_sample_node.getNetwork_device());
					}
					insert_list.add(ambari_node);
				}
			}
			
			
			
			
			// 디비에 적재
			insert_lustre_info.setList(insert_list);
		}else {
			return false;
		}
		
		try {
			if(insert_lustre_nodes(insert_lustre_info)) {
				return true;
			}else {
				return false;
			}
		} catch (Exception e) {
			LOGGER.error(e.getMessage());
			return false;
		}
		
		
		
	}
	
	
	/**
	 * lustre list up
	 * @param insert_lustre_info
	 * @return
	 */
	public boolean insert_lustre_nodes(LustreNodesVO insert_lustre_info) {
		return sqlSession.insert("com.xiilab.lustre.api.insert_lustre_list",insert_lustre_info) > 0;
	}
	
	
	
	

	/**
	 * AMBARI Server 호스트명 가져오기
	 * @return AMBARI 호스트명
	 */
	public String getAmbariServerHostName() {
		
		HttpServletRequest request = ((ServletRequestAttributes)RequestContextHolder.currentRequestAttributes()).getRequest();
		String serverIP = request.getLocalAddr(); // 서버아이피
		
		String DEFAULT_AMBARI_SERVER_HOST = serverIP;
		
		String result = serverIP;
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
	 * 테이블 생성 여부 확인 및 테이블 생성
	 * @return
	 */
	public Map<String, Object> checkCreatedTables() {
		
		List<String> check_list = getTableList();
		Map<String, Object> result = new HashMap<>();
		
		try {
			if(!check_list.contains("lustre_nodes")) {
				LOGGER.info("not lustre_nodes table create table....");
				sqlSession.update("com.xiilab.lustre.api.create_lustre_nodes_table");
			}
			if(!check_list.contains("disk_info")) {
				LOGGER.info("not disk_info table create table....");
				sqlSession.update("com.xiilab.lustre.api.create_disk_info_table");
			}
			if(!check_list.contains("lustre_log")) {
				LOGGER.info("not lustre_log table create table....");
				sqlSession.update("com.xiilab.lustre.api.create_lustre_log_table");
			}
			if(!check_list.contains("lustre_file_system_list")) {
				LOGGER.info("not lustre_file_system_list table create table....");
				sqlSession.update("com.xiilab.lustre.api.create_lustre_file_system_list_table");
			}
			result.put("status", true);
		} catch (Exception e) {
			LOGGER.error(e.getMessage());
			result.put("status", false);
			result.put("log", e.toString());
		}
		return result;
	}
	
	/**
	 * 테이블 명가져오기
	 * @return
	 */
	public List<String> getTableList(){
		List<String> result = sqlSession.selectList("com.xiilab.lustre.api.check_lustre_nodes_table");
//		LOGGER.info("find table list {}",result);
		return result;
	}



	/**
	 * Lustre 정보 가져오기
	 * @return
	 */
	public List<LustreNodesVO> getLustreNodes(LustreNodesVO lustreNodesVO) {
		List<LustreNodesVO> result = sqlSession.selectList("com.xiilab.lustre.api.getLustreNodes",lustreNodesVO);
		for (int i = 0; i < result.size() ; i++) {
			LustreNodesVO temp = result.get(i);
			temp.setDisk_list(getDiskInfo(temp.getNum()));
		}
		return result;
	}

	/**
	 * Lustre 정보 가져오기 (fs_step 추가)
	 * @return
	 */
	public List<LustreNodesVO> getLustreNodesForFileSystem(LustreNodesVO lustreNodesVO) {
		List<LustreNodesVO> result = sqlSession.selectList("com.xiilab.lustre.api.getLustreNodesForFileSystem",lustreNodesVO);
		for (int i = 0; i < result.size() ; i++) {
			LustreNodesVO temp = result.get(i);
			temp.setDisk_list(getDiskInfo(temp.getNum()));
		}
		return result;
	}

	
	/**
	 * 디스크 정보 가져오기
	 * @param LustreNodesKey
	 * @return
	 */
	public List<DiskInforVO> getDiskInfo(Long LustreNodesKey){
		DiskInforVO diskInforVO =  new DiskInforVO();
		diskInforVO.setLustre_nodes_key(LustreNodesKey);
		return sqlSession.selectList("com.xiilab.lustre.api.getDiskInfo",diskInforVO);
	}



	/**
	 * Lustre Node 업데이트
	 * @param lustre_info
	 */
	public boolean setLustreNodes(LustreNodesVO lustre_info) {
		return sqlSession.update("com.xiilab.lustre.api.setLustreNodes",lustre_info) > 0;
	}



	/**
	 * OST 마직막 번호 찾기
	 * @param diskType 
	 * @return
	 */
	public int getDiskLastIndex(String diskType) {
		Integer result = sqlSession.selectOne("com.xiilab.lustre.api.getOstLastIndex",diskType);
		return result != null ? result : -1;
	}

	/**
	 * lustre node 중에서 마직막 키값 찾기
	 * @param file_system_num 
	 * @param string
	 * @return
	 */
	public int getNodeLastIndex(String nodeType, Long file_system_num) {
		Map<String, Object> search_info = new HashMap<>();
		search_info.put("node_type", nodeType);
		search_info.put("file_system_num", file_system_num);
		
		Integer result = sqlSession.selectOne("com.xiilab.lustre.api.getNodeLastIndex",search_info);
		return result != null ? result : -1;
	}


	/**
	 * 디스크 정보 저장
	 * save disk information
	 * @param diskInforVO
	 * @return
	 */
	public boolean save_disk_info(DiskInforVO diskInforVO) {
		return sqlSession.insert("com.xiilab.lustre.api.save_disk_info",diskInforVO) > 0;
	}



	/**
	 * 로그저장
	 * @param save_log_rowKey
	 * @param line
	 * @param type
	 * @param type 
	 * @param hostname 
	 */
	public boolean saveCommandLog(String save_log_rowKey, String line, String log_label, String type, String hostname) {
		LustreLogVO lustreLogVO = new LustreLogVO();
		lustreLogVO.setData(line);
		lustreLogVO.setLog_type(type);
		lustreLogVO.setRow_key(save_log_rowKey);
		lustreLogVO.setLog_label(log_label);
		lustreLogVO.setHost_name(hostname);
		
		return sqlSession.insert("com.xiilab.lustre.api.saveCommandLog",lustreLogVO) > 0;
	}



	/**
	 * row_key 기준으로 로그목록 불려오기
	 * @return
	 */
	public List<LustreLogVO> getLogList() {
		return sqlSession.selectList("com.xiilab.lustre.api.getLogList");
	}



	/**
	 * row_key 을 이용하여 로그 내용보기
	 * @param lustreLogVO
	 * @return
	 */
	public List<LustreLogVO> viewLog(LustreLogVO lustreLogVO) {
		return sqlSession.selectList("com.xiilab.lustre.api.viewLog",lustreLogVO);
	}



	/**
	 * num 기준으로 num 보다 큰 로그내용 출력
	 * @param lustreLogVO
	 * @return
	 */
	public List<LustreLogVO> viewLastLogLine(LustreLogVO lustreLogVO) {
		return sqlSession.selectList("com.xiilab.lustre.api.viewLastLogLine",lustreLogVO);
	}



	/**
	 * 노드의 디스크 정보 읽어오기
	 * @param diskInforVO
	 * @return
	 */
	public List<DiskInforVO> getDisks(DiskInforVO diskInforVO) {
		return sqlSession.selectList("com.xiilab.lustre.api.getDiskInfo",diskInforVO);
	}
	
	



	/**
	 * 디스크 정보 갱신
	 * @param diskInfo
	 * @return
	 */
	public boolean updateDisk(DiskInforVO diskInfo) {
		return sqlSession.update("com.xiilab.lustre.api.updateDisk", diskInfo) > 0;
	}



	/**
	 * lustre.conf 파일 읽어오기
	 * @param lustreNodesVO
	 * @return
	 */
	public List<Map<String, Object>> readLustreConf(LustreNodesVO lustreNodesVO) {
		List<Map<String, Object>> result = new ArrayList<>();
		List<LustreNodesVO> lustre_list = getLustreNodes(lustreNodesVO);
		String command = "cat /etc/modprobe.d/lustre.conf";
		for (LustreNodesVO lustre_node : lustre_list) {
			Map<String, Object> tmpMap = new HashMap<>();
			Map<String, Object> resultMap = sshClient.execjsch(lustre_node, command);
			tmpMap.put("host_name", lustre_node.getHost_name());
			tmpMap.put("status", (Boolean) resultMap.get("status"));
			tmpMap.put("node_type", lustre_node.getNode_type());
			tmpMap.put("network_device", lustre_node.getNetwork_device());
			
			String data = (String) resultMap.get("log");
			
			// 마직막 개행문자 제거
			StringBuilder b = new StringBuilder(data);
			b.replace(data.lastIndexOf("\n"), data.lastIndexOf("\n") + 1, "" );
			data = b.toString();
			
			tmpMap.put("data", data);
			result.add(tmpMap);
		}
		
		return result;
	}



	/**
	 * get client folder
	 * @param fs_num 
	 * @return
	 */
	public String getClientFolder(Long fs_num) {
		return sqlSession.selectOne("com.xiilab.lustre.api.getClientFolder",fs_num);
	}


	/**
	 * delete disk for table
	 * @param diskInforVO
	 */
	public boolean deleteDisk(DiskInforVO diskInforVO) {
		return sqlSession.delete("com.xiilab.lustre.api.deleteDisk", diskInforVO) > 0;
	}


	/**
	 * read file system list
	 * @return
	 */
	public List<LustreFileSystemListVO> getFsList() {
		List<LustreFileSystemListVO> result = sqlSession.selectList("com.xiilab.lustre.api.getFsList");
		return result;
	}
	
	
	/**
	 * read file system list
	 * @return
	 */
	public List<LustreFileSystemListVO> searchLustreFileSystemList(LustreFileSystemListVO lustreFileSystemListVO) {
		List<LustreFileSystemListVO> result = sqlSession.selectList("com.xiilab.lustre.api.searchLustreFileSystemList",lustreFileSystemListVO);
		return result;
	}
	
	/**
	 * add file system
	 * @param file_system
	 * @return
	 */
	public Boolean addFileSystem(LustreFileSystemListVO file_system) {
		Integer result = sqlSession.insert("com.xiilab.lustre.api.addFileSystem",file_system);
		return result > 0;
	}

	/**
	 * shkim 20181214 : 파일시스템 중복 체크  
	 * 18.12.27 je.kim num or fs_name
	 * checkFileSystem
	 * @param file_system
	 * @return
	 */
	public int checkFileSystem(LustreFileSystemListVO file_system) {
		return sqlSession.selectOne("com.xiilab.lustre.api.checkFileSystem",file_system);
	}

	/**
	 * view file system
	 * @param file_system
	 * @return
	 */
	public LustreFileSystemListVO viewFileSystem(LustreFileSystemListVO file_system) {
		return sqlSession.selectOne("com.xiilab.lustre.api.viewFileSystem",file_system);
	}


	/**
	 * modify file system
	 * @param file_system
	 * @return
	 */
	public Boolean setFileSystem(LustreFileSystemListVO file_system) {
		return sqlSession.update("com.xiilab.lustre.api.setFileSystem",file_system) > 0;
	}


	/**
	 * 1개 이상 step 2 로 진행된 파일시스템이 있는지 검사하는 메서드
	 * @return
	 */
	public Boolean isSetLustreConf() {
		return sqlSession.selectList("com.xiilab.lustre.api.isSetLustreConf").size() < 1;
	}

	/**
	 * shkim 20190108
	 * Lustre node_type로 num을 가져옴  
	 * @return
	 */
	public long getLustreTypeNum(DiskInforVO disk_info) {
		
		String result = sqlSession.selectOne("com.xiilab.lustre.api.getLustreTypeNum", disk_info);
		
		return Long.parseLong(result);
	}

	
	/**
	 * shkim 20190109
	 * MGT disk_name를 가져오는 메소드   
	 * @return
	 */
	public String getMGTDisk_Name(DiskInforVO disk_info) {
		
		String result = sqlSession.selectOne("com.xiilab.lustre.api.getMGTDisk_Name", disk_info);
		
		return result;
	}
	
	/**
	 * shkim 20190109
	 * MGT 설치 유무를 확인하는 메소드     
	 * @return
	 */
	public int checkMGT() {
		
		int result = sqlSession.selectOne("com.xiilab.lustre.api.checkMGT");
		
		return result;
	}


	/**
	 * 삭제된 디스크 정보들을 찾는 메서드
	 * @param diskInforVO
	 * @return
	 */
	public List<DiskInforVO> is_ost_remove(DiskInforVO diskInforVO) {
		return sqlSession.selectList("com.xiilab.lustre.api.isOstRemove",diskInforVO);
	}


	/**
	 * remove lustre filesystem
	 * @param file_system
	 * @return
	 */
	public Boolean removeLustreFilesystem(LustreFileSystemListVO file_system) {
		return sqlSession.update("com.xiilab.lustre.api.removeLustreFilesystem",file_system) > 0;
	}
	
}

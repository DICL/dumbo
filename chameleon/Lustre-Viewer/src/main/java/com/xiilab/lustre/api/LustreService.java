package com.xiilab.lustre.api;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TimeZone;

import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.collections4.Predicate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import com.xiilab.lustre.model.DiskInforVO;
import com.xiilab.lustre.model.LustreFileSystemListVO;
import com.xiilab.lustre.model.LustreLogVO;
import com.xiilab.lustre.model.LustreNodesVO;
import com.xiilab.lustre.utilities.IdGeneration;
import com.xiilab.lustre.utilities.SshClient;

/**
 * @author xiilab
 *
 */
@Service
public class LustreService {
	
	static final Logger LOGGER = LoggerFactory.getLogger(LustreService.class);
	
	static final String LUSTRE_FS_NAME = "mylustre";
	static final String MDT_MOUNT_NAME = "/mnt/mdt";
	static final String OST_MOUNT_NAME = "/mnt/ost";
	
	// MDT 추가시 진행될 단계
	static final int ADD_MDT_STEP = 2;
	// OST 추가시 진행될 단계
	static final int ADD_OST_STEP = 3;
	// CLIENT 추가시 진행될 단계
	static final int ADD_CLIENT_STEP = 4;
	// 완료된 단계
	static final int COMPLETE_STEP = 5;
	
	private final static String MDS_SETTING_LABEL = "MDS Setting";
	private final static String ADD_MGT_LABEL = "Add MGT Disk";
	private final static String OSS_SETTING_LABEL = "OSS Setting";
	private final static String CLIENT_SETTING_LABEL = "Client Setting";
	
	@Autowired
	private LustreDAO lustreDAO;
	
	@Autowired
	private SshClient sshClient;

	/**
	 * ambari stack 가져오기
	 * @return
	 */
	public String getNodeMatricDataArr() {
		return lustreDAO.getNodeMatricDataArr();
	}
	
	
	
	/**
	 * ambari api로 호출된 내용을 토대로 러스터 서비스가 설치된 노드를 탐색하는 메서드
	 * @return
	 */
	public List<LustreNodesVO> findLustreNodesForAmbari(){
		return lustreDAO.findLustreNodesForAmbari();
	}
	
	
	/**
	 * 디비에 적재되어 있는 러스터정보을 읽어오고 없으면 암바리를 통하여 동기화 하는 메서드
	 * @param file_system 
	 * @return
	 */
	public boolean syncLustreTable(LustreFileSystemListVO file_system) {
		return lustreDAO.syncLustreTable(file_system);
	}
	
	
	/**
	 * 기본테이블 생성 여부 확인
	 * @return
	 */
	public Map<String, Object> checkCreatedTables() {
		return lustreDAO.checkCreatedTables();
	}



	/**
	 * Lustre 정보 가져오기
	 * @return
	 */
	public List<LustreNodesVO> getLustreNodes(LustreNodesVO lustreNodesVO) {
		return lustreDAO.getLustreNodes(lustreNodesVO);
	}

	/**
	 * 디스크 추가하기
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @return
	 */
//	public Map<String, Object> addDIsk(LustreNodesVO lustreNodesVO, DiskInforVO diskInforVO) {
//		Map<String, Object> result =  new HashMap<>();
//		
//		// LustreNodesVO 입력여부 검사
//		if(
//				lustreNodesVO.getNode_type() == null
//				|| "".equals(lustreNodesVO.getNode_type())
//				|| lustreNodesVO.getIndex() == null
//			) {
//			result.put("status", false);
//			result.put("log", "input lustre host");
//			return result;
//		}
//		
//		
//		// LustreNodes 가져오기
//		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
//		// 만약에 내용이 없으면 종료
//		if(lustreNodes.size() <= 0) {
//			result.put("status", false);
//			result.put("log", "not find lustre ingormation");
//			return result;
//		}
//		LustreNodesVO lustre_info = lustreNodes.get(0);
//		
//		// 만약에 네트워크 명이 있으면 업데이트
//		if(lustreNodesVO.getNetwork_device() != null && !"".equals(lustreNodesVO.getNetwork_device())) {
//			lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
//			lustreDAO.setLustreNodes(lustre_info);
//		}
//		
//		// 러스터 커널 설정(네트워크 설정)
//		result = setLustreConf(lustre_info);
//		
//		// 해당 노드가 MDS 인지 OSS 인지 판별
//		if(lustre_info.getNode_type().equals("MDS")) {
//			result = addMDTDisk(lustre_info,diskInforVO);
//		}
//		
//		
//		if(lustre_info.getNode_type().equals("OSS")) {
//			result = addOSTDisk(lustre_info,diskInforVO);
//		}
//		
//		
//		return result;
//	}


	
	/**
	 * OST 추가
	 * @param lustre_info
	 * @param diskInforVO
	 * @return 
	 */
//	public Map<String, Object> addOSTDisk(LustreNodesVO lustre_info, DiskInforVO diskInforVO) {
//		Map<String, Object> result =  new HashMap<>();
//		
//		int ost_index = getDiskLastIndex("OST") + 1;
//		
//		// add index number
//		diskInforVO.setIndex(ost_index);
//		diskInforVO.setDisk_type("OST");
//		diskInforVO.setLustre_nodes_key(lustre_info.getNum());
//		
//		// ost disk format
//		result = ost_disk_format(lustre_info,diskInforVO);
//		
//		
//		// save database ost information
//		save_disk_info(diskInforVO);
//		
//		return result;
//	}



	/**
	 * database save disk information
	 * @param lustre_info
	 * @param diskInforVO
	 */
	private boolean save_disk_info(DiskInforVO diskInforVO) {
		return lustreDAO.save_disk_info(diskInforVO);
	}



	/**
	 * ost disk format
	 * @param oss_lustre_info
	 * @param diskInforVO
	 * @return
	 */
//	private Map<String, Object> ost_disk_format(LustreNodesVO oss_lustre_info, DiskInforVO diskInforVO) {
//		Map<String, Object> result =  new HashMap<>();
//		
//		// MDS 정보 가져오기
//		LustreNodesVO search_mds_node = new LustreNodesVO();
//		search_mds_node.setNode_type("MDS");
//			
//		// LustreNodes 가져오기
//		List<LustreNodesVO> lustreNodes = getLustreNodes(search_mds_node);
//		// 만약에 내용이 없으면 종료
//		if(lustreNodes.size() <= 0) {
//			result.put("status", false);
//			result.put("log", "not find lustre ingormation");
//			return result;
//		}
//		LustreNodesVO mds_lustre_info = lustreNodes.get(0);
//				
//				
//				
//		String mds_host_name = mds_lustre_info.getHost_name();
//		String mds_network_device_name = mds_lustre_info.getNetwork_device();
//		String oss_target_disk = diskInforVO.getDisk_name();
//		int ost_index = diskInforVO.getIndex();
//		
//				
//				
//		// 포맷 명령어
//		String ost_format_command = "mkfs.lustre --fsname="+LUSTRE_FS_NAME+"  --ost --mgsnode="+mds_host_name+"@"+mds_network_device_name
//				+" --index=" + ost_index + " --reformat " + oss_target_disk;
//				
//		result = sshClient.execjsch(oss_lustre_info, ost_format_command , false);
//		
//		
//		return result;
//	}



	/**
	 * OST 마직막 번호 찾기
	 * @param DiskType 
	 * @return
	 */
	private int getDiskLastIndex(String DiskType) {
		return lustreDAO.getDiskLastIndex(DiskType);
	}



	/**
	 * MDT 추가
	 * @param lustre_info
	 * @param diskInforVO
	 * @return 
	 */
//	public Map<String, Object> addMDTDisk(LustreNodesVO mds_lustre_info, DiskInforVO diskInforVO) {
//		Map<String, Object> result =  new HashMap<>();
//		
//		int mdt_index = getDiskLastIndex("MDT") + 1;
//		
//		// add index number
//		diskInforVO.setIndex(mdt_index);
//		diskInforVO.setDisk_type("MDT");
//		diskInforVO.setLustre_nodes_key(mds_lustre_info.getNum());
//		
//		// ost disk format
//		result = mdt_disk_format(mds_lustre_info,diskInforVO);
//		
//		
//		// save database ost information
//		save_disk_info(diskInforVO);
//		
//		return result;
//	}



	/**
	 * MDT 디스크 포맷
	 * @param mds_lustre_info
	 * @param diskInforVO
	 * @return
	 */
//	private Map<String, Object> mdt_disk_format(LustreNodesVO mds_lustre_info, DiskInforVO diskInforVO) {
//		Map<String, Object> result =  new HashMap<>();
//		
//		int mdt_index = diskInforVO.getIndex();
//		String mds_target_disk = diskInforVO.getDisk_name();
//		// 포맷 명령어
//		String mdt_format_command = "mkfs.lustre --fsname="+LUSTRE_FS_NAME+"  --mdt --mgs --index="+mdt_index+" " + mds_target_disk ;
//		
//		result = sshClient.execjsch(mds_lustre_info, mdt_format_command , false);
//		
//		return result;
//	}



	/**
	 * lustre.conf 생성 및 커널 모듈 올려주기 (네트워크 설정)
	 * @param lustre_info 
	 * @param server
	 * @return
	 */
//	public Map<String, Object> setLustreConf(LustreNodesVO lustre_info){
//		Map<String, Object> result =  new HashMap<>();
//		
//		String result_text = "";
//		
//		String network_name = lustre_info.getNetwork_device();
//		
//		//  네트워크 명이 지정안되어 있으면 종료
//		if(network_name == null || "".equals(network_name)) {
//			result.put("status", false);
//			result.put("log", "not input network name");
//			return result;
//		}
//		
//		String Lustrecontent = "options lnet networks=\""+network_name+"\"";
//		String conf_file_path = "/etc/modprobe.d/lustre.conf";
//		
//		String[] command_list = {
//				"sed '$d' "+conf_file_path+" > "+conf_file_path+".tmp",
//				"echo \""+Lustrecontent+"\" >> "+conf_file_path+".tmp",
//				"mv "+conf_file_path+".tmp "+conf_file_path+""
//			};
//		
//		
//		// lustre.conf 생성 명령어
//		for (String createConfigFileCommand : command_list) {
//			result = sshClient.execjsch(lustre_info, createConfigFileCommand, false);
//			result_text += result.get("log");
//		}
//		
//		// 커널 모듈 올려주는 명령어
//		String ModprobeLustreCommand = "modprobe lustre";
//		
//		result = sshClient.execjsch(lustre_info, ModprobeLustreCommand, false);
//		result_text += result.get("log");
//		
//		result.put("log", result_text);
//		
//		return result;
//	}



	/**
	 * 네트워크 디바이스명 가져오기
	 * @param lustreNodesVO 
	 * @return
	 */
	public List<String> getNetWorkDeviceList_bak(LustreNodesVO lustreNodesVO) {
		List<String> result = new ArrayList<>();
		
		// LustreNodes 가져오기
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			LOGGER.error("getNetWorkDevice - Not lustre nodes");
			return result;
		}
		LustreNodesVO lustre_info = lustreNodes.get(0);
		String command = "ip addr show | grep '^[0-9]: [0-9|a-z]*'";
		
		Map<String, Object> ssh_result = sshClient.execjsch(lustre_info, command);
		if((boolean) ssh_result.get("status")) {
			String log_data = (String) ssh_result.get("log");
			String[] log_list = log_data.split(System.getProperty("line.separator"));
			for (String log_line : log_list) {
				String[] temp_string = log_line.split("\\s");
				if(temp_string.length > 1) {
					result.add(temp_string[1].replaceAll(":", ""));
				}
			}
		}
		
		
		
		return result;
	}
	
	/**
	 * 네트워크 디바이스명 가져오기 - 디바이스 타입도 가져오기
	 * @param lustreNodesVO
	 * @return
	 */
	public List<String> getNetWorkDeviceList(LustreNodesVO lustreNodesVO) {
		List<String> result = new ArrayList<>();
		
		// LustreNodes 가져오기
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			LOGGER.error("getNetWorkDevice - Not lustre nodes");
			return result;
		}
		LustreNodesVO lustre_info = lustreNodes.get(0);
		// 1 .ip addr show 명령어 실행
		// 2. 숫자 : 숫자 또는 문자 또는 'link/' 으로 시작되는 문자열 만 추출
		// 3. 공백을 기준으로 첫번쨰와 두번째 문자열만 추출
		// 4. 앞글자가 숫자
		// ip addr show | grep '^[0-9]: [0-9|a-z]*\|link/' | awk '{ print $1 } { print $2 }' | grep '^[^0-9]' | awk ' NR % 2 == 1 { printf "%s", $0 } NR % 2 == 0 { print $0 } '
		String command = "ip addr show | grep '^[0-9]: [0-9|a-z]*\\|link/' | awk '{ print $1 } { print $2 }' | grep '^[^0-9]' | awk ' NR % 2 == 1 { printf \"%s\", $0 } NR % 2 == 0 { print $0 } '";
		
		Map<String, Object> ssh_result = sshClient.execjsch(lustre_info, command);
		if((boolean) ssh_result.get("status")) {
			String log_data = (String) ssh_result.get("log");
			
			String[] log_list = log_data.split(System.getProperty("line.separator"));
			for (String log_line : log_list) {
				result.add(log_line);
			}
		}
		return result;
	}



	/**
	 * 전체 노드의 네트워크 디바이스 가져오기
	 * @return
	 */
	public Map<String, Object> getNetWorkDevice(LustreNodesVO lustreNodesVO) {
		Map<String, Object> result = new HashMap<>();
		
		List<LustreNodesVO> node_list = getLustreNodes(lustreNodesVO);
		for (LustreNodesVO lustre_node : node_list) {
			result.put(lustre_node.getHost_name(), getNetWorkDeviceList(lustre_node));
		}
		
		
		return result;
	}



	/**
	 * 전체 노드의 디스크 정보 가져오기
	 * @param lustreNodesVO
	 * @return
	 */
	public Map<String, Object> getDiskDevice(LustreNodesVO lustreNodesVO) {
		Map<String, Object> result = new HashMap<>();
		List<LustreNodesVO> node_list = getLustreNodes(lustreNodesVO);
		for (LustreNodesVO lustre_node : node_list) {
			result.put(lustre_node.getHost_name(), getDiskDeviceList(lustre_node));
		}
		
		return result;
	}
	
	/**
	 * 전체 노드의 디스크 정보 가져오기
	 * @param lustreNodesVO
	 * @return
	 */
	public Map<String, Object> new_getDiskDevice(LustreNodesVO lustreNodesVO) {
		Map<String, Object> result = new HashMap<>();
		List<LustreNodesVO> node_list = getLustreNodes(lustreNodesVO);
		for (LustreNodesVO lustre_node : node_list) {
			result.put(lustre_node.getHost_name(), new_getDiskDeviceList(lustre_node));
		}
		
		return result;
	}

	
	
	
	
	/**
	 * 암바리을 통하여 MDS Node 정보를 가져오는 메서드
	 * @return
	 */
	public LustreNodesVO getAmbariForMDSnode() {
		// 암바리에서 노드정보 가져오기
		List<LustreNodesVO> node_list = findLustreNodesForAmbari();
		
		Collection<LustreNodesVO> mds_node_list = CollectionUtils.select(node_list, new Predicate<Object>() {
			public boolean evaluate(Object a) {
				return ((LustreNodesVO) a).getNode_type().equals("MDS");
			}
		});
		if(mds_node_list.isEmpty()) {
			return null;
		}
		LustreNodesVO mds_node = mds_node_list.iterator().next();
		mds_node.setSsh_port(22);
		mds_node.setUser_id("root");
		mds_node.setPrivate_key(".ssh/id_rsa");
		
		return mds_node;
	}
	
	
	
	
	/**
	 * file system mgt 노드 추가시 디스크정보 가져오기
	 * @return
	 */
	public Map<String, Object> getAmbariForMDSDisks() {
		Map<String, Object> result = new HashMap<>();
		// MDS 노드 가져오기
		LustreNodesVO mds_node = getAmbariForMDSnode();
		// 190408 je.kim 수정
		result.put(mds_node.getHost_name(), new_getDiskDeviceList(mds_node));
		
		return result;
	}



	/**
	 * 해당노드의 디스크 정보 리스트 가져오기
	 * @param lustre_node
	 * @return
	 */
	private List<Map<String, Object>> getDiskDeviceList(LustreNodesVO lustre_node) {
		List<Map<String, Object>> result = new ArrayList<>();
//		String command = "fdisk -l | grep /dev | grep -Ev 'Disk'";
		// 190316 je.kim command 수정
		String command = "fdisk -l | grep /dev | grep Disk | sed -e 's/Disk//g' | awk '{print $1}' | sed -e 's/://g'";
		
		Map<String, Object> ssh_result = sshClient.execjsch(lustre_node, command);
		if((boolean) ssh_result.get("status")) {
			String log_data = (String) ssh_result.get("log");
			String[] log_list = log_data.split(System.getProperty("line.separator"));
			for (String log_line : log_list) {
				Map<String, Object> tmp_map = new HashMap<>();
				String disk_name = log_line.trim();
				
				if(disk_name.contains("mapper")) continue;
				
				tmp_map.put("name", disk_name);
				tmp_map.put("size", getDiskSize(lustre_node,disk_name));
				//tmp_map.put("info", getDiskPatitionInfo(lustre_node,disk_name));
				result.add(tmp_map);
			}
			
			
//			String log_data = (String) ssh_result.get("log");
//			String[] log_list = log_data.split(System.getProperty("line.separator"));
//			for (String log_line : log_list) {
//				String[] temp_string = log_line.split("\\s");
//				if(temp_string.length > 1) {
//					Map<String, Object> tmp_map = new HashMap<>();
//					String disk_name = temp_string[0];
//					tmp_map.put("name", disk_name);
//					tmp_map.put("size", getDiskSize(lustre_node,disk_name));
//					
//					result.add(tmp_map);
//				}
//			}
		}
		
		
		
		return result;
	}

	
	/**
	 * 해당노드의 디스크 정보 리스트 가져오기
	 * @param lustre_node
	 * @return
	 */
	private List<Map<String, Object>> new_getDiskDeviceList(LustreNodesVO lustre_node) {
		List<Map<String, Object>> result = new ArrayList<>();
//		String command = "fdisk -l | grep /dev | grep -Ev 'Disk'";
		// 190316 je.kim command 수정
		// lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT -ds | awk '$5 == "" {print $1,$2,$3}' | awk '$3 == "disk" {print $1,$2}'
		String command = "lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT -ds | awk '$5 == \"\" {print $1,$2,$3}' | awk '$3 == \"disk\" {print $1,$2}'";
		
		Map<String, Object> ssh_result = sshClient.execjsch(lustre_node, command);
		if((boolean) ssh_result.get("status")) {
			String log_data = (String) ssh_result.get("log");
			String[] log_list = log_data.split(System.getProperty("line.separator"));
			for (String log_line : log_list) {
				Map<String, Object> tmp_map = new HashMap<>();
				if(!log_line.equals("")) {
					String[] temp_string = log_line.split("\\s");
					if(temp_string.length > 0) {
						String disk_name = "/dev/" + temp_string[0];
						tmp_map.put("name", disk_name);
					}
					if(temp_string.length > 1) {
						tmp_map.put("size", temp_string[1]);
					}
					result.add(tmp_map);
				}
			}
		}
	
		return result;
	}
	
	


//	private String getDiskPatitionInfo(LustreNodesVO lustre_node, String disk_name) {
//		String result = "";
//		String command = "file -sL "+disk_name+" ";
//		Map<String, Object> ssh_result = sshClient.execjsch(lustre_node, command);
//		result =  ( (String) ssh_result.get("log")).replace("\n", "");
//		
//		return result;
//	}



	/**
	 * 디스크 사이즈 구하기
	 * @param lustre_node 
	 * @param disk_name
	 * @return
	 */
	private String getDiskSize(LustreNodesVO lustre_node, String disk_name) {
		String result = "";
		String command = "fdisk "+disk_name+" -l | grep Disk ";
		Map<String, Object> ssh_result = sshClient.execjsch(lustre_node, command);
		if((boolean) ssh_result.get("status")) {
			String log_data = (String) ssh_result.get("log");
			String[] log_list = log_data.split(System.getProperty("line.separator"));
			for (String log_line : log_list) {
				String[] temp_string = log_line.split("\\s");
				if(temp_string.length > 3) {
					result = temp_string[2] + temp_string[3].replaceAll(",", "");
					break;
				}
			}
		}
		
		return result;
	}

	/**
	 * lustre.conf 생성 및 커널 모듈 올려주기 (네트워크 설정) -비동기
	 * @param lustre_info 
	 * @param server
	 * @return 
	 */
	public Boolean setLustreConf(LustreNodesVO lustre_info, String row_key, String log_label) {
		boolean result = false;
		
		String network_name = lustre_info.getNetwork_device();
		//  네트워크 명이 지정안되어 있으면 종료
		if(network_name == null || "".equals(network_name)) {
			return false;
		}
		
		String Lustrecontent = "options lnet networks=\\\"tcp("+network_name+")\\\"";
		String conf_file_path = "/etc/modprobe.d/lustre.conf";
		
		String[] command_list = {
				"sed '$d' "+conf_file_path+" > "+conf_file_path+".tmp",
				"echo \""+Lustrecontent+"\" >> "+conf_file_path+".tmp",
				"mv "+conf_file_path+".tmp "+conf_file_path+""
			};
		
		// lustre.conf 생성 명령어
		for (String createConfigFileCommand : command_list) {
			result = (boolean) sshClient.execjschForLogging(lustre_info, createConfigFileCommand, row_key, log_label).get("status");
		}
				
		// 커널 모듈 올려주는 명령어
		String ModprobeLustreCommand = "modprobe lustre";
		result = (boolean) sshClient.execjschForLogging(lustre_info, ModprobeLustreCommand, row_key , log_label).get("status");
		
		return result;
	}



	/**
	 * MDT 디스크 추가 - 비동기
	 * @param mds_lustre_info mds 정보
	 * @param diskInforVO 디스크 정보
	 * @param lustre_fs  파일시스템정보
	 * @param row_key 유니크한 키값
	 * @param mdsSettingLabel 이름
	 */
	public Boolean addMDTDisk(LustreNodesVO mds_lustre_info, DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs, String row_key, String mdsSettingLabel) {
		// 181227 je.kim mdt는 한개임
		int mdt_index = getDiskLastIndex("MDT") + 1;
//		int mdt_index = 0;
		// add index number
		diskInforVO.setIndex(mdt_index);
		diskInforVO.setDisk_type("MDT");
		diskInforVO.setLustre_nodes_key(mds_lustre_info.getNum());
		
		// 단계변경
		lustre_fs.setFs_step(ADD_MDT_STEP);
		lustre_fs.setOrigin_fs_name(lustre_fs.getFs_name());
		
		boolean tmp_check_logic = mdt_disk_format(mds_lustre_info,diskInforVO,lustre_fs,row_key,mdsSettingLabel);
		if(tmp_check_logic) {
			mdt_disk_mount(mds_lustre_info,diskInforVO,lustre_fs,row_key,mdsSettingLabel);
			if(checkMountDisk(mds_lustre_info,diskInforVO.getDisk_name())) {
				
				// file system 점검 및 디스크 정보 적재
				// setFileSystem 에 러스터 노드들을 적재하는 메서드가있
				return setFileSystem(lustre_fs) && save_disk_info(diskInforVO) ;
			}else {
				return false;
			}
		}else {
			return false;
		}
	}
	
	
	/**
	 * 마운트 확인
	 * @param diskName 폴더명
	 * @return
	 */
	public Boolean checkMountDisk(LustreNodesVO lustre_info,String diskName) {
		Boolean result = false;
		String command = "mount | grep " + diskName + "";
		
		Map<String, Object> resultMap = sshClient.execjsch(lustre_info, command);
		if( ((String) resultMap.get("log")).contains(diskName) ) {
			result = true;
		}else {
			result = false;
		}
		
		
		return result;
	}
	
	
	/**
	 * file system list 을 가져오면서 마운트 여부도 체크하는 메서드
	 * @return
	 */
	public List<LustreFileSystemListVO> getFileSystemIsMountList() {
		List<LustreFileSystemListVO> result = lustreDAO.getFsList();
		
		
		for (int i = 0; i < result.size(); i++) {
			LustreFileSystemListVO tmp = result.get(i);
			LOGGER.info("file system {} is step : {}",tmp.getFs_name(),tmp.getFs_step());
			// 만약 설정이 완료가 안되어 있다면 넘어감
			if(tmp.getFs_step() < 5) {
				LOGGER.info("file system {} is not complete",tmp.getFs_name());
				continue;
			}
			
			// CLIENT 노드만 가져오기
			LustreNodesVO search_lustre_node = new LustreNodesVO();
			search_lustre_node.setFile_system_num(tmp.getNum()); // file system 설정
			search_lustre_node.setNode_type("CLIENT"); // 타입은 CLIENT
			List<LustreNodesVO> client_list = getLustreNodes(search_lustre_node);
			
			tmp.setIs_client_mount(true); // 해당 luster mount 여부는 참으로 설정
			for(LustreNodesVO client_info : client_list) {
				// 폴더명 삽입
				tmp.setLustre_client_folder(client_info.getLustre_client_folder());
				// client 노드들을 탐색하면서 하나라도 mount 가 안되어 있다면 거짓으로 설정후에 강제중단
				if(! checkMountDisk(client_info , client_info.getLustre_client_folder())) {
					tmp.setIs_client_mount(false);
					break;
				}
			}
			
			// 해당내용 적용
			result.set(i, tmp);
		}
		
		return result;
	}
	
	
	/**
	 * lustre.conf 파일을 읽어와서 tcp 인지 o2ib1 인지 식별하는 메서드
	 * @param lustre_info
	 * @return tcp or o2ib1
	 */
	public String getLustreConfNetworkType(LustreNodesVO lustre_info) {
		String command = "cat /etc/modprobe.d/lustre.conf";
		Map<String , Object> result = sshClient.execjsch(lustre_info, command);
		
		String ssh_result = (String) result.get("log");
		
		if(ssh_result.contains("o2ib1")) {
			return "o2ib1";
		}else if(ssh_result.contains("tcp")) {
			return "tcp";
		}
		return null;
	}
	
	/**
	 * /etc/hosts 파일을 읽어와서 host name 을 통하여 ip 을 찾는 메서드
	 * @param lustre_info
	 * @return
	 */
	public String getHostIP(LustreNodesVO lustre_info) {
		
		String host_name = lustre_info.getHost_name();
		String command = "cat /etc/hosts | grep "+host_name+" | awk '{print $1}'";
		
		Map<String , Object> result = sshClient.execjsch(lustre_info, command);
		String ssh_result = (String) result.get("log");
		
		// 개행문자를 제서하여 출력
		return ssh_result.replaceAll("(\r\n|\r|\n|\n\r)", "");
	}
	
	
	
	/**
	 * lustre client 폴더 마운트
	 * @param luste_file_system
	 * @param row_key
	 * @return
	 */
	public Boolean mountClientFolder(LustreFileSystemListVO luste_file_system, String row_key) {
		
		String title = "Mount Client Folder";
		String command = "";
		
		boolean result = true;
		Map<String,Object> temp_result = new HashMap<>();
		
		
		
		// CLIENT 노드만 가져오기
		LustreNodesVO search_lustre_node = new LustreNodesVO();
		search_lustre_node.setFile_system_num(luste_file_system.getNum()); // file system 설정
		search_lustre_node.setNode_type("CLIENT"); // 타입은 CLIENT
		List<LustreNodesVO> client_list = getLustreNodes(search_lustre_node);
		
		
		// MDS 노드정보 가져오기
		search_lustre_node.setNode_type("MDS"); // 타입은 MDS
		List<LustreNodesVO> mds_list = getLustreNodes(search_lustre_node);
		LustreNodesVO mds_info = null;
		
		if(mds_list.size() > 0) {
			mds_info = mds_list.get(0);
		}else {
			return false;
		}
		
		
		for(LustreNodesVO client_info : client_list) {
			String network_name = getLustreConfNetworkType(client_info);
			String client_folder = client_info.getLustre_client_folder();
			
			// je.kim get mds ip
			// String mds_ip = getHostIP(mds_info);
			String mds_ip = get_MDS_IP_Address(mds_info.getNetwork_device());
			
			
			// 190219 je.kim 호스트별로 구분
			row_key = IdGeneration.generateId(16);
			
			command = "mkdir -p " + client_folder;
			temp_result = sshClient.execjschForLogging(client_info, command, row_key, title);
			result = (boolean) temp_result.get("status");
			
			
			command = "mount -t lustre "+mds_ip+"@"+network_name+":/"+luste_file_system.getFs_name()+" "+client_folder;
			temp_result = sshClient.execjschForLogging(client_info, command, row_key, title);
			result = (boolean) temp_result.get("status");
		}
		
		return result;
	}



	/**
	 * lustre client 폴더 언마운트
	 * @param luste_file_system
	 * @param row_key
	 * @return
	 */
	public Boolean umountClientFolder(LustreFileSystemListVO luste_file_system, String row_key) {
		
		String title = "UMount Client Folder";
		
		boolean result = true;
		Map<String,Object> temp_result = new HashMap<>();
		String command = "";
		
		
		// CLIENT 노드만 가져오기
		LustreNodesVO search_lustre_node = new LustreNodesVO();
		search_lustre_node.setFile_system_num(luste_file_system.getNum()); // file system 설정
		search_lustre_node.setNode_type("CLIENT"); // 타입은 CLIENT
		List<LustreNodesVO> client_list = getLustreNodes(search_lustre_node);
		
		
		
		for(LustreNodesVO client_info : client_list) {
			// 190411 je.kim 노드별로 구분
			String tmp_row_key = IdGeneration.generateId(16);
			
			// 190404 je.kim client 폴더 유마운트 전에 tmp2 폴더 마운트 확인후에 tmp2 폴더 유마운트
			// tmp2 마운트 확인
			if(checkMountDisk(client_info , "/tmp2")) {
				String umount_tmp2_label = "umount tmp2 Folder";
				command = "fuser -ck /tmp2";
				result = (boolean)sshClient.execjschForLogging(client_info, command, tmp_row_key, umount_tmp2_label).get("status");
				command = "umount -l /tmp2";
				result = (boolean)sshClient.execjschForLogging(client_info, command, tmp_row_key, umount_tmp2_label).get("status");
			}
			
			
			
			String client_folder = client_info.getLustre_client_folder();
			// 190404 je.kim 'fuser -ck' add command  
			if(checkMountDisk(client_info , client_folder)) {
				command = "fuser -ck "+client_folder;
				result = (boolean)sshClient.execjschForLogging(client_info, command, tmp_row_key, title).get("status");
			}
			
			command = "umount " + client_folder;
			temp_result = sshClient.execjschForLogging(client_info, command, tmp_row_key, title);
			result = (boolean) temp_result.get("status");
		}
		
		return result;
	}



	/**
	 * MDT 디스크 마운트 - 비동기
	 * @param mds_lustre_info - MDS 정보
	 * @param diskInforVO - 디스크정보
	 * @param lustre_fs  파일시스템 네임
	 * @param row_key - 유니크한 키값
	 * @param mdsSettingLabel - 이름
	 * @return
	 */
	private boolean mdt_disk_mount(LustreNodesVO mds_lustre_info, DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs, String row_key, String mdsSettingLabel) {
		
		String device_name = diskInforVO.getDisk_name();
		
		// 디렉토리 생성 명령어
		//int mdt_index = getDiskLastIndex("MDT") + 1;
		int mdt_index =diskInforVO.getIndex();
		// String mdt_directory_name = MDT_MOUNT_NAME+mdt_index;
		// shkim 20190115 MDT index 0으로 고정  
		String mdt_directory_name =  "/mnt/"+lustre_fs.getFs_name()+"/"+"mdt0";
		String create_directory_command = "mkdir -p "+mdt_directory_name;
		sshClient.execjschForLogging(mds_lustre_info, create_directory_command, row_key , mdsSettingLabel);
		
		// 마운트 명령어
		String mdt_mount_command = "mount -t lustre "+device_name+" "+mdt_directory_name;
		Map<String, Object> result = sshClient.execjschForLogging(mds_lustre_info, mdt_mount_command, row_key , mdsSettingLabel);
		
		return (boolean) result.get("status");
	}



	/**
	 * MDT 디스크 포맷 - 비동기
	 * @param mds_lustre_info - mdt 정보
	 * @param diskInforVO  - 디스크 정보
	 * @param lustre_fs  - 파일시스템정보
	 * @param row_key - 유니크한 키값
	 * @param mdsSettingLabel - 이름
	 * @return
	 */
	private boolean mdt_disk_format(LustreNodesVO mds_lustre_info, DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs, String row_key, String mdsSettingLabel) {
		
		int mdt_index = diskInforVO.getIndex();
		String mds_target_disk = diskInforVO.getDisk_name();
		
		// 190315 je.kim host name -> IP Address
		
		String mds_ip_addr = get_MDS_IP_Address(mds_lustre_info.getNetwork_device());
		
		// 190403 je.kim ifconfig 이 설치가 안되어 있으면 설치중단
		if(mds_ip_addr == null) {
			return false;
		}
		
		//String mds_host_name = mds_lustre_info.getHost_name();
		String file_system_name = lustre_fs.getFs_name();
		// 포맷 명령어
		// 181227 je.kim fs name 연동
		//String mdt_format_command = "mkfs.lustre --fsname="+LUSTRE_FS_NAME+"  --mdt --mgs --index="+mdt_index+" --reformat " + mds_target_disk ;
		//String mdt_format_command = "mkfs.lustre --fsname="+file_system_name+"  --mdt --mgs --index="+mdt_index+" --reformat " + mds_target_disk ;
		// 190107 je.kim mgs 연동
		String mdt_format_command = "mkfs.lustre --fsname="+file_system_name+"  --mdt --mgsnode="+mds_ip_addr+"@tcp --index=0 --reformat " + mds_target_disk ;
		
		Map<String, Object> result = sshClient.execjschForLogging(mds_lustre_info, mdt_format_command, row_key , mdsSettingLabel);
		
		return (boolean) result.get("status");
	}



	/**
	 * OST 디스크 추가 - 비동기
	 * @param lustre_info - OSS 정보
	 * @param disk_list -디스크 정보
	 * @param lustre_fs - file system info
	 * @param row_key - 유니크한 키
	 * @param ossSettingLabel - 이름
	 */
	public Boolean addOSTDisk(LustreNodesVO lustre_info, List<DiskInforVO> disk_list, LustreFileSystemListVO lustre_fs, String row_key, String ossSettingLabel) {
		//int ost_index = getDiskLastIndex("OST") + 1;
		// add index number
		boolean result = false;
		
		for (DiskInforVO disk : disk_list) {
			if(disk.getIndex() == null) {
				//disk.setIndex(ost_index);
				//je.kim 190215 사용자가 이젠 수동으로 ost 번호를 할당하므로 필요없음
				LOGGER.error("Not OST NUMBER EXIT");
				return false;
			}
			disk.setDisk_type("OST");
			disk.setLustre_nodes_key(lustre_info.getNum());
			
			if(disk.getRemove_disk_frag() == null || !disk.getRemove_disk_frag()) {
				// ost 포맷
				result = ost_disk_format(lustre_info,disk,lustre_fs,row_key,ossSettingLabel,true);
				// ost 마운트
				result = ost_disk_mount(lustre_info,disk,lustre_fs,row_key,ossSettingLabel);
				
				
				
				// 마운트 확인
				if(checkMountDisk(lustre_info,disk.getDisk_name())) {
					if (is_ost_remove(disk,lustre_fs)) {
						// 190215 je.kim 만약 해당디스크가 과거에 삭제되었다면 해당 디스크정보을 업데이트 처리
						
						disk.setIs_remove(false);
						disk.setDisk_type("OST");
						disk.setFile_system_num(lustre_fs.getNum());
						
						List<DiskInforVO> remove_disk_list = lustreDAO.is_ost_remove(disk);
						DiskInforVO set_disk = remove_disk_list.get(0);
						
						set_disk.setIs_remove(true);
						set_disk.setDisk_name(disk.getDisk_name());
						set_disk.setLustre_nodes_key(lustre_info.getNum());
						
						
						//set_disk.setIs_activate(false);
						
						String temp_row_key = IdGeneration.generateId(16);
						setMaxcount("20000",lustre_info, set_disk, lustre_fs, temp_row_key, "Active OST");
						// ative 처리
						activate(lustre_info, set_disk, lustre_fs, temp_row_key, "Active OST");
						
						
						result = updateDisk(set_disk);
					}else {
						result = save_disk_info(disk);
					}
				}
				
				//ost_index ++;
			}
		}
		
		// 단계변경
		// 190215 je.kim 완료되었다면 무시하고 넘어감
		if(result && lustre_fs.getFs_step() != COMPLETE_STEP) {
			lustre_fs.setFs_step(ADD_OST_STEP);
			lustre_fs.setOrigin_fs_name(lustre_fs.getFs_name());
			result = setFileSystem(lustre_fs);
		}
		
		return result;
	}



	



	/**
	 * OST 디스크 마운트 - 비동기
	 * @param lustre_info OSS 정보
	 * @param diskInforVO - 디스크 정보
	 * @param lustre_fs 
	 * @param row_key - 유니크 키값
	 * @param ossSettingLabel - 이름
	 * @return
	 */
	private boolean ost_disk_mount(LustreNodesVO lustre_info, DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs, String row_key, String ossSettingLabel) {
		
		int index = diskInforVO.getIndex();
		String device_name = diskInforVO.getDisk_name();
		
		// file system 이
		String file_system_name = lustre_fs.getFs_name();
		
		// 디렉토리 생성 명령어
//		String create_directory_command = "mkdir "+OST_MOUNT_NAME+index;
		String ost_directory_name =  "/mnt/"+lustre_fs.getFs_name()+"/"+"ost"+index;
		String create_directory_command = "mkdir -p "+ost_directory_name;
		
		sshClient.execjschForLogging(lustre_info, create_directory_command, row_key , ossSettingLabel);		
		
		// 마운트 명령어
		String ost_mount_command = "mount -t lustre "+device_name+" "+ost_directory_name;
		Map<String, Object> result  = sshClient.execjschForLogging(lustre_info, ost_mount_command, row_key ,ossSettingLabel);
		
		return (boolean) result.get("status");
	}


	/**
	 * 과거에 해당 ost 을 삭제했었는지 확인하는 메서드
	 * @param diskInforVO disc info
	 * @param lustre_fs lustre file system info
	 * @return
	 * true : 과거에 삭제함 , false 삭제안함
	 */
	private boolean is_ost_remove(DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs) {
		
		if(diskInforVO.getDisk_name() == null || "".equals(diskInforVO.getDisk_name())) {
			LOGGER.warn("method is_ost_remove not set disk_name");
			return false;
		}
		
		diskInforVO.setDisk_type("OST");
		diskInforVO.setIs_remove(false);
		diskInforVO.setFile_system_num(lustre_fs.getNum());
		List<DiskInforVO> disk_list = lustreDAO.is_ost_remove(diskInforVO);
		
		if(disk_list.size() > 0) {
			LOGGER.info("method is_ost_remove disk_name \"{}\" is deleted",disk_list.get(0).getDisk_name());
			return true;
		}else {
			LOGGER.warn("method is_ost_remove find disks \"{}\" wrong search",disk_list.size());
			return false;
		}
		
		
	}
	

	/**
	 * OST 디스크 포맷 - 비동기
	 * @param lustre_info OSS 정보
	 * @param diskInforVO 디스크 정보
	 * @param lustre_fs 
	 * @param row_key 유니크한 키값
	 * @param ossSettingLabel 이름
	 * @param check_disk true : disk 을 조회하여 과거에 삭제했는지 확인 , false 무시하고 진행
	 * 
	 * @return
	 */
	private boolean ost_disk_format(
			LustreNodesVO lustre_info, 
			DiskInforVO diskInforVO, 
			LustreFileSystemListVO lustre_fs, 
			String row_key, 
			String ossSettingLabel,
			boolean check_disk
			) {
		
		// MDS 정보 가져오기
		LustreNodesVO search_mds_node = new LustreNodesVO();
		search_mds_node.setNode_type("MDS");
		
		// LustreNodes 가져오기
		List<LustreNodesVO> lustreNodes = getLustreNodes(search_mds_node);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO mds_lustre_info = lustreNodes.get(0);		
		
		// 190315 je.kim hostname -> ip address
		//String mds_host_name = mds_lustre_info.getHost_name();
		String mds_ip_adderss = get_MDS_IP_Address(mds_lustre_info.getNetwork_device());
		
		
//		String mds_network_device_name = mds_lustre_info.getNetwork_device();
		String oss_target_disk = diskInforVO.getDisk_name();
		int ost_index = diskInforVO.getIndex();
		
		// file system 이
		String file_system_name = lustre_fs.getFs_name();
		
		// 190215 je.kim 과거에 디스크를 삭제했는지 확인하고 만약 과거에 디스크를 삭제 했다면 해당 디스크에 --replace 옵션을 추가
		String replace_commend = (is_ost_remove(diskInforVO,lustre_fs) && check_disk) ? " --replace " : "";
		
		// 포맷 명령어
//		String ost_format_command = "mkfs.lustre --fsname="+LUSTRE_FS_NAME+"  --ost --mgsnode="+mds_host_name+"@"+mds_network_device_name
		//String ost_format_command = "mkfs.lustre --fsname="+LUSTRE_FS_NAME+"  --ost --mgsnode="+mds_host_name+"@"+"tcp"
		// mkfs.lustre --fsname=lustre1 --ost --mgsnode=node05@tcp --index=2 --reformat --replace /dev/sdg1
		String ost_format_command = "mkfs.lustre --fsname="+file_system_name+"  --ost --mgsnode="+mds_ip_adderss+"@"+"tcp"
				+" --index=" + ost_index + " --reformat " +replace_commend + oss_target_disk;	
		
		Map<String, Object> result  = sshClient.execjschForLogging(lustre_info, ost_format_command, row_key,ossSettingLabel);
		
		return (boolean) result.get("status");
	}
	
	
	/**
	 * OST 제거
	 * @param ost_info
	 * @param file_system 
	 * @param disk
	 * @param remove_ost_row_key
	 */
	public boolean removeOSTDisk(LustreNodesVO ost_info, DiskInforVO diskInforVO, LustreFileSystemListVO file_system, String remove_ost_row_key) {
		boolean status = false;
		Map<String, Object> tmp_command_result = new HashMap<>();
		
		// MDS 정보 가져오기
		LustreNodesVO search_mds_node = new LustreNodesVO();
		search_mds_node.setNode_type("MDS");
						
		// MDS LustreNodes 가져오기
		List<LustreNodesVO> lustreNodes = getLustreNodes(search_mds_node);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO mds_lustre_info = lustreNodes.get(0);
		
		
		// Client Lustre Node 가져오기
		LustreNodesVO search_client_node = new LustreNodesVO();
		search_client_node.setNode_type("CLIENT");
		search_client_node.setFile_system_num(file_system.getNum());
		List<LustreNodesVO> client_node_list = getLustreNodes(search_client_node);
		
		int ost_index = diskInforVO.getIndex();
		// 190111 je.kim 10진수 -> 16진수로 변
		String ost_suffix = String.format("%04x", ost_index); // 0005
		
		//String mdt_fsname = lustreDAO.getMdtFsname();
		String mdt_fsname = file_system.getFs_name();
		
		
		
		// mylustre-OST0005
		String ost_name = mdt_fsname+"-OST"+ost_suffix;
		
		// mylustre-OST0003-osc-MDT0000
		String osc_name = mdt_fsname+"-OST"+ost_suffix+"-osc-MDT0000";
		
		// Deactivate the OSC on the MDS nodes
		// [mds] lctl set_param osp.osc_name.max_create_count=0
		String deactivate_radom_key = IdGeneration.generateId(16);
		String deactivate_osc_mds_command = "lctl set_param osp."+osc_name+".max_create_count=0";
		// Deactivate the OSC on the MDS nodes label
		String deactivate_osc_mds_label = "Deactivate the "+osc_name+" on the MDS nodes";
		tmp_command_result  = sshClient.execjschForLogging(mds_lustre_info, deactivate_osc_mds_command, deactivate_radom_key,deactivate_osc_mds_label);
		status = (boolean) tmp_command_result.get("status");
		
		
		// Migrate data to other OSTs in the file system
		// [client] lfs find --ost ost_name /mount/point | lfs_migrate -y 
//		String client_mount_point1 = getClientFolder(file_system.getNum()); // get client folder
//		String client_mount_point2 = "/tmp2"; // default mount point;
		String migrate_data_label = "Migrate data to other "+ost_name+" in the client file system";
		
		for (LustreNodesVO client_node : client_node_list) {
			String migrate_data_radom_key = IdGeneration.generateId(16);
			String client_mount_point1 = client_node.getLustre_client_folder();
			
			// 190401 je.kim pass lfs_migrate tmp2
//			String migrate_data_command2 = "lfs find --ost "+ost_name+" "+client_mount_point2+" | lfs_migrate -y";
//			tmp_command_result  = sshClient.execjschForLogging(client_node, migrate_data_command2, migrate_data_radom_key, migrate_data_label);
//			status = (boolean) tmp_command_result.get("status");

			// 190411 find client mount folder and mount after break loop
			if(checkMountDisk(client_node, client_mount_point1)) {
				String migrate_data_command1 = "lfs find --ost "+ost_name+" "+client_mount_point1+" | lfs_migrate -y";
				tmp_command_result  = sshClient.execjschForLogging(client_node, migrate_data_command1, migrate_data_radom_key, migrate_data_label);
				status = (boolean) tmp_command_result.get("status");
				break;
			}
		}
		
		// Deactivate the OST
		String deactivate_command = "lctl conf_param "+ost_name+".osc.active=0";
		String deactivate_ost_label = "OST"+(diskInforVO.getIndex())+" Deactivate";
		tmp_command_result  = sshClient.execjschForLogging(mds_lustre_info, deactivate_command, IdGeneration.generateId(16) ,deactivate_ost_label);
		status = (boolean) tmp_command_result.get("status");
		
		// umount disk
		String ost_mount_point = "/mnt/"+mdt_fsname+"/ost"+ost_index;
		String umaont_command = "umount "+ost_mount_point;
		String ost_umount_label = "Umount OST"+ost_index;
		tmp_command_result  = sshClient.execjschForLogging(ost_info, umaont_command, remove_ost_row_key ,ost_umount_label);
		status = (boolean) tmp_command_result.get("status");
		
		// delete ost
		// 19.02.25 je.kim 디스크는 삭제안함
		// lustreDAO.deleteDisk(diskInforVO);
		
		// 삭제대신 삭재 플래그를 삽입하여 업데이트 처리
		diskInforVO.setIs_remove(false);
		lustreDAO.updateDisk(diskInforVO);
		
		
		return status;
	}
	
	
	/**
	 * max counter 수정
	 * @param max_create_count : max count
	 * @param ostInfo : lustre node
	 * @param diskInforVO : disk info
	 * @param lustre_fs : lustre file system
	 * @param row_key : random key
	 * @param ossSettingLabel : command name
	 */
	private boolean setMaxcount(
			String max_create_count,
			LustreNodesVO ostInfo, 
			DiskInforVO diskInforVO, 
			LustreFileSystemListVO lustre_fs,
			String row_key, 
			String ossSettingLabel
			) {
		// MDS 정보 가져오기
		LustreNodesVO search_mds_node = new LustreNodesVO();
		search_mds_node.setNode_type("MDS");
		
		List<LustreNodesVO> lustreNodes = getLustreNodes(search_mds_node);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO mds_lustre_info = lustreNodes.get(0);
		
		int ost_index = diskInforVO.getIndex();
		// 190111 je.kim 10진수 -> 16진수로 변경
		String ost_suffix = String.format("%04x", ost_index); // 0005
		
		//String mdt_fsname = lustreDAO.getMdtFsname();

		// FS name 가져오기
		String mdt_fsname =lustre_fs.getFs_name();
				
		// mylustre-OST0005
		String osc_name = mdt_fsname+"-OST"+ost_suffix+"-osc-MDT0000";
		
		// set max_create_count 커맨드
		// ex) lctl set_param osp.lustre0-OST0005-osc-MDT0000.max_create_count=0
		// lctl set_param osp.lustre0-OST0005-osc-MDT0000.max_create_count=0
		String deactivate_command = "lctl set_param osp."+osc_name+".max_create_count="+max_create_count;
				
				
		Map<String, Object> result  = sshClient.execjschForLogging(mds_lustre_info, deactivate_command, row_key,ossSettingLabel);
		return (boolean) result.get("status");
	}
	
	
	
	/**
	 * activate 커맨드 전송
	 * 190107 je.kim file system 추가
	 * @param ostInfo
	 * @param diskInforVO
	 * @param lustre_fs 
	 * @param row_key
	 * @param ossSettingLabel
	 * @return
	 */
	public boolean activate(LustreNodesVO ostInfo, DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs, String row_key, String ossSettingLabel) {
		// MDS 정보 가져오기
		LustreNodesVO search_mds_node = new LustreNodesVO();
		search_mds_node.setNode_type("MDS");
				
		// LustreNodes 가져오기
		List<LustreNodesVO> lustreNodes = getLustreNodes(search_mds_node);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO mds_lustre_info = lustreNodes.get(0);
		
		
		
		int ost_index = diskInforVO.getIndex();
		// 190111 je.kim 10진수 -> 16진수로 변경
		String ost_suffix = String.format("%04x", ost_index); // 0005
		
		//String mdt_fsname = lustreDAO.getMdtFsname();

		// FS name 가져오기
		String mdt_fsname =lustre_fs.getFs_name();
		
		// mylustre-OST0005
		String ost_name = mdt_fsname+"-OST"+ost_suffix;
		
		// activate 커맨드
		// ex) lctl conf_param mylustre-OST0005.osc.active=1
		String deactivate_command = "lctl conf_param "+ost_name+".osc.active=1";
		
		
		Map<String, Object> result  = sshClient.execjschForLogging(mds_lustre_info, deactivate_command, row_key,ossSettingLabel);
		return (boolean) result.get("status");
	}
	


	/**
	 * deactivate 커맨드 전송
	 * 190108 je.kim lustre file system 추가
	 * @param ostInfo
	 * @param lustre_fs 
	 * @param diskInfo
	 * @param row_key
	 * @return
	 */
	public boolean deactivate(LustreNodesVO ostInfo, DiskInforVO diskInforVO, LustreFileSystemListVO lustre_fs, String row_key, String ossSettingLabel) {
		
		// MDS 정보 가져오기
		LustreNodesVO search_mds_node = new LustreNodesVO();
		search_mds_node.setNode_type("MDS");
		
		
				
		// LustreNodes 가져오기
		List<LustreNodesVO> lustreNodes = getLustreNodes(search_mds_node);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO mds_lustre_info = lustreNodes.get(0);
		
		
		
		int ost_index = diskInforVO.getIndex();
		
		// 190111 je.kim 10진수 -> 16진수로 변경
		String ost_suffix = String.format("%04d", ost_index); // 0005
		
		//String mdt_fsname = lustreDAO.getMdtFsname();
		// FS name 가져오기
		String mdt_fsname =lustre_fs.getFs_name();
		
		// mylustre-OST0005
		String ost_name = mdt_fsname+"-OST"+ost_suffix;
		
		// deactivate 커맨드
		// ex) lctl conf_param mylustre-OST0005.osc.active=0
		String deactivate_command = "lctl conf_param "+ost_name+".osc.active=0";
		
		
		Map<String, Object> result  = sshClient.execjschForLogging(mds_lustre_info, deactivate_command, row_key,ossSettingLabel);
		return (boolean) result.get("status");
	}
	
	
	/**
	 * LNET Network Start
	 * @param lustreNode
	 * @param Lustrecontent
	 * @param Label
	 * @param row_key
	 * @return
	 */
	public boolean networkStart(LustreNodesVO lustreNode, String Lustrecontent, LustreFileSystemListVO lustre_fs, String Label, String row_key) {
		boolean result = true;
		
		// mds node info
		LustreNodesVO mds_search_node = new LustreNodesVO();
		mds_search_node.setNode_type("MDS");
		List<LustreNodesVO> mds_node_list = getLustreNodes(mds_search_node);
		LustreNodesVO mds_node = mds_node_list.get(0);
		
		String conf_file_path = "/etc/modprobe.d/lustre.conf";
		String[] command_list = {
				"echo \""+Lustrecontent.replaceAll("\"", "\\\\\"")+"\" >> "+conf_file_path+".tmp",
				"mv "+conf_file_path+".tmp "+conf_file_path+"",
				"modprobe -v lnet",
				"lctl network configure",
			};
		
		
		
		for (int i = 0; i < command_list.length; i++) {
			String command = command_list[i];
			Map<String, Object> tmp_map  = sshClient.execjschForLogging(lustreNode, command, row_key, Label);
			if((boolean) tmp_map.get("status")) {
				result = false;
			}
		}
		
		
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setHost_name(lustreNode.getHost_name());
		search_node.setNode_type(lustreNode.getNode_type());
		
		List<LustreNodesVO> node_list = lustreDAO.getLustreNodesForFileSystem(search_node);
		
		
		// 추가 동작
		switch (lustreNode.getNode_type()) {
			case "CLIENT":
//				String mds_node_name = mds_node.getHost_name();
				
				String mds_node_ip = get_MDS_IP_Address(mds_node.getNetwork_device());
				
				for (LustreNodesVO lustre_node : node_list) {
					//String tmp_rowkey = IdGeneration.generateId(16);
					
					// 마운트를 위하 파일시스템 정보들을 가져오기
					LustreFileSystemListVO search_file_system = new LustreFileSystemListVO();
					search_file_system.setNum(lustre_node.getFile_system_num());
					LustreFileSystemListVO lustre_file_system = lustreDAO.viewFileSystem(search_file_system);
					String fs_name = lustre_file_system.getFs_name();

					// 해당노드들의 마운트 폴터 탐색
					String client_folder = lustre_node.getLustre_client_folder();
					String command  = "mount -t lustre "+mds_node_ip+"@tcp:/"+fs_name+" "+client_folder;
					sshClient.execjschForLogging(lustreNode, command, row_key, Label);
				}
				
				break;
//			case "MDS": case "OSS":
//				List<DiskInforVO> disk_list = lustreNode.getDisk_list();
//				for (DiskInforVO disk_info : disk_list) {
//					// /mnt/ost1 ...
//					
//					String mount_point = "/mnt/"+fs_name+"/" +disk_info.getDisk_type().toLowerCase() + disk_info.getIndex();
//					String command = "mount -t lustre " + disk_info.getDisk_name() + " " + mount_point;
//					sshClient.execjschForLogging(lustreNode, command, row_key, Label);
//				}
//				break;
			default:
				result = false;
				break;
		}
		
		
		return result;
		
	}
	
	
	/**
	 * 모든 파일시스템에 마운트 된 폴더들을 언마운트후에 네트워크 스탑 (클라이언트만 가능)
	 * @param lustreNodesVO
	 * @param label
	 * @param row_key
	 */
	public boolean new_networkStop(LustreNodesVO lustreNode, LustreFileSystemListVO lustre_fs, String Label, String row_key) {
		boolean result = true;
		
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setHost_name(lustreNode.getHost_name());
		search_node.setNode_type(lustreNode.getNode_type());
		List<LustreNodesVO> client_list = lustreDAO.getLustreNodesForFileSystem(search_node);
		
		switch (lustreNode.getNode_type()) {
		
		case "CLIENT":
					
			String command = "lctl --net tcp conn_list";
			result = (boolean)sshClient.execjschForLogging(lustreNode, command, row_key, Label).get("status");
			
			// tmp2 마운트 확인
			if(checkMountDisk(lustreNode , "/tmp2")) {
				command = "fuser -ck /tmp2";
				result = (boolean)sshClient.execjschForLogging(lustreNode, command, row_key, Label).get("status");
				
				command = "umount -l /tmp2";
				result = (boolean)sshClient.execjschForLogging(lustreNode, command, row_key, Label).get("status");
			}
			
			
			
			// 같은호스트내의 다른 러스터 파일시스템을 확인해서 유마운트
			// ex) node01(client) => /lustre0 , /lustre1 ......
			for (LustreNodesVO client_node : client_list) {
				String client_folder = client_node.getLustre_client_folder();
				//String tmp_rowkey = IdGeneration.generateId(16);
				// 러스터 클아이언트 마운트 확인
				if(checkMountDisk(lustreNode , client_folder)) {
					command = "fuser -ck " + client_folder;
					result = (boolean) sshClient.execjschForLogging(lustreNode, command, row_key, Label).get("status");
					
					command = "umount -l " + client_folder;
					result = (boolean) sshClient.execjschForLogging(lustreNode, command, row_key, Label).get("status");
				}
			}
			
			
			break;
		}
	
		
		String[] command_list = {
				"lctl network unconfigure",
				"lustre_rmmod",
		};
		for (int i = 0; i < command_list.length; i++) {
			String command = command_list[i];
			Map<String, Object> tmp_map  = sshClient.execjschForLogging(lustreNode, command, row_key, Label);
			if((boolean) tmp_map.get("status")) {
				result = false;
			}
		}
		return result;
	}


	/**
	 * LNET Network Stop - 미사용
	 * @param lustreNode
	 * @param label
	 * @param row_key
	 */
	public boolean networkStop(LustreNodesVO lustreNode, LustreFileSystemListVO lustre_fs, String Label, String row_key) {
		boolean result = true;
		
		String fs_name = lustre_fs.getFs_name();
		String clint_mount_point = lustreDAO.getClientFolder(lustre_fs.getNum());
		
		switch (lustreNode.getNode_type()) {
			case "CLIENT":
				
				String[] add_clien_command_list = {
						"lctl --net tcp conn_list",
						
						"fuser -ck /tmp2",
						"umount -l /tmp2",
						
						"fuser -ck "+ clint_mount_point,
						"umount -l "+ clint_mount_point,
				};
				for (int i = 0; i < add_clien_command_list.length; i++) {
					String command = add_clien_command_list[i];
					sshClient.execjschForLogging(lustreNode, command, row_key, Label);
				}
				break;
			case "MDS": case "OSS":
				List<DiskInforVO> disk_list = lustreNode.getDisk_list();
				for (DiskInforVO disk_info : disk_list) {
					
					String command = "lctl --net tcp conn_list";
					sshClient.execjschForLogging(lustreNode, command, row_key, Label);
					
					// /mnt/ost1 ...
					String mount_point = "/mnt/"+fs_name+"/" + disk_info.getDisk_type().toLowerCase() + disk_info.getIndex();
					
					command = "fuser -ck "+mount_point;
					sshClient.execjschForLogging(lustreNode, command, row_key, Label);
					
					
					command = "umount " + mount_point;
					sshClient.execjschForLogging(lustreNode, command, row_key, Label);
				}
				break;
			default:
				break;
		}
		
		
		String[] command_list = {
				"lctl network unconfigure",
				"lustre_rmmod",
		};
		for (int i = 0; i < command_list.length; i++) {
			String command = command_list[i];
			Map<String, Object> tmp_map  = sshClient.execjschForLogging(lustreNode, command, row_key, Label);
			if((boolean) tmp_map.get("status")) {
				result = false;
			}
		}
		return result;
	}



	/**
	 * 로그 리스트 가져오기
	 * @return
	 */
	public List<LustreLogVO> getLogList() {
		return lustreDAO.getLogList();
	}



	/**
	 * 로그내용 보기
	 * @param lustreLogVO
	 * @return
	 */
	public List<LustreLogVO> viewLog(LustreLogVO lustreLogVO) {
		return lustreDAO.viewLog(lustreLogVO);
	}



	/**
	 * 최근 로그내용 보기
	 * @param lustreLogVO
	 * @return
	 */
	public List<LustreLogVO> viewLastLogLine(LustreLogVO lustreLogVO) {
		return lustreDAO.viewLastLogLine(lustreLogVO);
	}



	/**
	 * 디비에 있는 디스크정보 가져오기
	 * @param diskInforVO
	 * @return
	 */
	public List<DiskInforVO> getDisks(DiskInforVO diskInforVO) {
		return lustreDAO.getDisks(diskInforVO);
	}



	/**
	 * 디스크 정보 업데이트
	 * @param diskInfo
	 * @return
	 */
	public boolean updateDisk(DiskInforVO diskInfo) {
		return lustreDAO.updateDisk(diskInfo);
	}



	/**
	 * 각 노드의 lustre.conf 파일 읽어오기
	 * @param lustreNodesVO
	 * @return
	 */
	public List<Map<String, Object>> readLustreConf(LustreNodesVO lustreNodesVO) {
		return lustreDAO.readLustreConf(lustreNodesVO);
	}



	/**
	 * 클라이언트 폴더 출력
	 * @return
	 */
	public String getClientFolder(Long fs_num) {
		return lustreDAO.getClientFolder(fs_num);
	}
	
	/**
	 * ost 디스크 백업 - spring async 사용안함
	 * @param disk 백업할 디스크정보
	 * @param lustre_fs 러스터 파일시스템 이름
	 * @param row_key
	 * @return
	 */
	public Boolean no_async_backupDisk(DiskInforVO disk, LustreFileSystemListVO lustre_fs, String row_key) {
		// 저장된 디스크 정보 가져오기
		List<DiskInforVO> disk_list = getDisks(disk);
		
		// 저장된 노드 정보 가져오기
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setHost_name(disk.getHost_name());
		List<LustreNodesVO> node_list = getLustreNodes(search_node);
		
		// 만약 검색된 데이터가 없을경우 예외처리
		if(disk_list.size() == 0 || node_list.size() == 0) {
			new Exception("not found disk information");
		}
		
		DiskInforVO disk_info = disk_list.get(0);
		// 백업될 파일 위치
		disk_info.setBackup_file_localtion(disk.getBackup_file_localtion());
		LustreNodesVO node_info = node_list.get(0);
		
		// 라벨정의
		String Label = "Backup " + disk_info.getDisk_name();

		
		return backupDisk(disk_info,node_info,lustre_fs,row_key,Label);
	}
	
	/**
	 * ost 디스크 복구 - spring async 사용안함
	 * @param disk 복구한 디스크 정보
	 * @param lustre_fs 러스트 파일시스템 이름
	 * @param row_key 
	 * @return
	 */
	public Boolean no_async_restoreDisk(DiskInforVO disk, LustreFileSystemListVO lustre_fs, String row_key) {
		
		// 저장된 노드 정보 가져오기
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setHost_name(disk.getHost_name());
		List<LustreNodesVO> node_list = getLustreNodes(search_node);
		
		// 디비에서 가져오기
    	lustre_fs = viewFileSystem(lustre_fs);
    	if(lustre_fs == null) {
    		return false;
    	}
		
		// 만약 검색된 데이터가 없을경우 예외처리
		if(node_list.size() == 0) {
			LOGGER.error("not found node information");
			return false;
			//new Exception("not found disk information");
		}
		LustreNodesVO node_info = node_list.get(0);

		
		disk.setLustre_nodes_key(node_info.getNum());
		
		// 라벨정의
		String Label = "Restore "+disk.getDisk_type()+disk.getIndex()+" For " + disk.getDisk_name();
		
		
		return restoreDisk(disk,node_info,lustre_fs,row_key,Label);
	}
	
	
	/**
	 * Restore Disk
	 * @param disk_info
	 * @param node_info
	 * @param lustre_fs 
	 * @param row_key
	 * @param label
	 * @return
	 */
	public Boolean restoreDisk(DiskInforVO disk_info, LustreNodesVO node_info, LustreFileSystemListVO lustre_fs, String row_key, String label) {
		boolean result = false;
		String command = "";
		
		// 0. deactivate disk
		if("OST".equals(disk_info.getDisk_type())) {
			// MDS 노드에서 수행되기 때문에 별도의 로우키 할당
			String temp_row_key = IdGeneration.generateId(16);
			String temp_deactivate_label = "Deactivate before disk recovery. (" + disk_info.getDisk_name() + ")";
			deactivate(node_info, disk_info, lustre_fs ,temp_row_key, temp_deactivate_label);
		}
		
		// 1. umount disk
		result = umountDisk(disk_info,node_info, lustre_fs, row_key, label);
		// 2. Format the new device
		if("OST".equals(disk_info.getDisk_type())) {
			result = ost_disk_format(node_info, disk_info,lustre_fs, row_key, label, true);
			// 190411 je.kim e2label 추가
			// e2label /dev/loop5 test0-OST0001
			String temp_disk_label = 
					lustre_fs.getFs_name() // lustre0
					+ "-"  // -
					+ disk_info.getDisk_type().toUpperCase()  // OST
					+ String.format("%04x", disk_info.getIndex()); // 0005
					;
			command = "e2label " + disk_info.getDisk_name() +" " + temp_disk_label;
			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		}else {
			// MDT일 경우 예외처리 (임시)
			LOGGER.error("Not Support MDT Disk");
			//new Exception("Not Support MDT Disk");
		}
		
		// 3. Create directory for mountpoint
		String temp_mount_point = "/mnt/" + ((disk_info.getDisk_type().equals("MDT")) ? "mdt" : "ost") + disk_info.getIndex();
		command = "mkdir -p " + temp_mount_point;
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		// 4. Mount the file system
		// mount -t ldiskfs /dev/sdd1 /mnt/ost6
		command = "mount -t ldiskfs " + disk_info.getDisk_name() + " " + temp_mount_point;
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		// 5. Change to the new file system mount point & Restore the file system backup
		// tar xzvpf /tmp/oss1-ost6-20181204.bak.tgz --xattrs --xattrs-include="trusted.*" --sparse
		// 194010 je.kim --warning=no-timestamp 추가
		command = "cd " + temp_mount_point + " && "+"tar xzvpf " + disk_info.getBackup_file_localtion() + " --xattrs --xattrs-include=\"trusted.*\" --sparse --warning=no-timestamp";
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		// 6. Verify that the extended attributes were restored
		// 190411 je.kim getattr, setattr 불필요
//		String[] backup_file_path_list = disk_info.getBackup_file_localtion().split("/");
//		// /tmp/oss1-ost6-20181204.bak.tgz -> oss1-ost6-20181204.bak
//		String temp_backup_file_name = backup_file_path_list[backup_file_path_list.length - 1].replaceAll(".tgz", "");
//		command = "cd " + temp_mount_point + " && "+"setfattr --restore="+temp_backup_file_name;
//		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		// 7. Remove old OI and LFSCK files
		command = "cd " + temp_mount_point + " && " + " rm -rf oi.16* lfsck_* LFSCK";
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		// Remove old CATALOGS (MDT only)
		// 190411 je.kim remove CATALOGS 
		//if("MDT".equals(disk_info.getDisk_type())) {
			command = "cd " + temp_mount_point + " && " + "rm -rf CATALOGS";
			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		//}
		
		// 8. Unmount the new file system
		command = "umount "+temp_mount_point;
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		// 9. Mount the target as lustre
		// result = mountDisk(disk_info,node_info,lustre_fs,row_key, label);
		// 190411 je.kim -o abort_recov 옵션 추가
		// mount -t lustre -o abort_recov /dev/loop5 /mnt/ost_1
		String file_system_name = lustre_fs.getFs_name();
		String new_mount_point = "/mnt/"+file_system_name+"/" + disk_info.getDisk_type().toLowerCase() + disk_info.getIndex(); 
		command = "mount -t lustre -o abort_recov " + disk_info.getDisk_name() + " " + new_mount_point;
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		if("OST".equals(disk_info.getDisk_type())) {
			// MDS 노드에서 수행되기 때문에 별도의 로우키 할당
			String temp_row_key = IdGeneration.generateId(16);
			String temp_deactivate_label = "Activate after disk recovery. (" + disk_info.getDisk_name() + ")";
			activate(node_info, disk_info, lustre_fs ,temp_row_key, temp_deactivate_label);
		}
		
		// 업데이트 할 디스크정보 생성하기
		
		
		// 디스크 정보를 업데이트 하기 위한 데이터 검색
		// 저장된 디스크 정보 가져오기
		
		DiskInforVO search_disk = new DiskInforVO();
		search_disk.setIndex(disk_info.getIndex());
		search_disk.setLustre_nodes_key(node_info.getNum());
		List<DiskInforVO> disk_list = getDisks(search_disk);
		
		// 만약 검색된 데이터가 없을경우 새로 디스크 정보를 만들기
//		if(disk_list.size() == 0) {
//			DiskInforVO save_disk = disk_info;
//			save_disk.setLustre_nodes_key(node_info.getNum());
//			// 10. create disk
//			result = save_disk_info(save_disk);
//		}else {
//			DiskInforVO  save_disk = disk_list.get(0);
//			// 디스크 명 교체
//			save_disk.setDisk_name(disk_info.getDisk_name());
//			// 10. update disk
//			result = updateDisk(save_disk);
//		}
		DiskInforVO tmp_disk = disk_info;
		
		if (is_ost_remove(tmp_disk,lustre_fs)) {
			tmp_disk.setIs_remove(false);
			tmp_disk.setDisk_type("OST");
			tmp_disk.setFile_system_num(lustre_fs.getNum());
			
			List<DiskInforVO> remove_disk_list = lustreDAO.is_ost_remove(disk_info);
			DiskInforVO set_disk = remove_disk_list.get(0);
			
			set_disk.setIs_remove(true);
			set_disk.setDisk_name(tmp_disk.getDisk_name());
			set_disk.setLustre_nodes_key(node_info.getNum());
			
			
			//set_disk.setIs_activate(false);
			
			String temp_row_key = IdGeneration.generateId(16);
			setMaxcount("20000",node_info, set_disk, lustre_fs, temp_row_key, "Active OST");
			// ative 처리
			activate(node_info, set_disk, lustre_fs, temp_row_key, "Active OST");
			
			
			result = updateDisk(set_disk);
			
		} 
		// 190410
		else if (disk_list.size() > 0) {
			DiskInforVO  save_disk = disk_list.get(0);
			// 디스크 명 교체
			save_disk.setDisk_name(disk_info.getDisk_name());
			// 10. update disk
			result = updateDisk(save_disk);
		}
		else {
			DiskInforVO save_disk = disk_info;
			save_disk.setLustre_nodes_key(node_info.getNum());
			// 10. create disk
			result = save_disk_info(save_disk);
		}
		
		
		
		
		
		return result;
	}

	
	
	/**
	 * Backup Disk
	 * @param disk_info
	 * @param node_info
	 * @param lustre_fs 
	 * @param row_key
	 * @param label
	 * @return
	 */
	public Boolean backupDisk(DiskInforVO disk_info, LustreNodesVO node_info, LustreFileSystemListVO lustre_fs, String row_key, String label) {
		boolean result = false;
		String command = "";
		
		// 1. umount disk
		result = umountDisk(disk_info,node_info,lustre_fs,row_key, label);
				
		// 2. Make a mountpoint for the file system (/mnt/tmp_ost0)
		// temp mount point folder
		String temp_mount_point = "/mnt/tmp_" + ((disk_info.getDisk_type().equals("MDT")) ? "mdt" : "ost") + disk_info.getIndex();
		command = "mkdir -p " + temp_mount_point;
		if(result) {
			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		}
		
		
		// 3. Mount the file system (ldiskfs)
		command = "mount -t ldiskfs "+disk_info.getDisk_name()+" "+temp_mount_point;
		if(result) {
			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		}
		
		// 4. Change to the mountpoint being backed up & Back up the extended attributes
		SimpleDateFormat printFormat = new SimpleDateFormat("yyyyMMdd");
		printFormat.setTimeZone(TimeZone.getTimeZone("Asia/Seoul"));
		
		// backup file name
		// oss1-ost1-181010.tar.gz
		String backup_file_name = 
				node_info.getNode_type().toLowerCase() + 
				node_info.getIndex() + "-"+
				disk_info.getDisk_type().toLowerCase() + 
				disk_info.getIndex() + "-"+
				printFormat.format(new Date())+".bak";
		
		// 190411 je.kim getattr, setattr 불필요
//		command = "cd " + temp_mount_point + " && " + " getfattr -R -d -m '.*' -e hex -P . > " + backup_file_name;
//		if(result) {
//			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
//		}
		
		// 5. Back up all file system data
		// 190401 je.kim file system name 기준으로 폴더를 생성하고 백업파일 위치
		String backup_file_localtion = disk_info.getBackup_file_localtion() + "/" + lustre_fs.getFs_name();
		
		// 190403 je.kim create backup location
		command = "mkdir -p " + backup_file_localtion;
		result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		
		command = "cd " + temp_mount_point + " && " + " tar czvf "+backup_file_localtion + "/" + backup_file_name+".tgz" + " --xattrs --xattrs-include=\"trusted.*\" --sparse .";
		if(result) {
			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		}
		
		// 6. Erase uncompressed files 
		// 190411 je.kim getattr, setattr 불필요
//		command = "cd " + temp_mount_point + " && " + "rm -f " + backup_file_name;
//		if(result) {
//			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
//		}		
		
		// 7. Unmount the file system
		command = "umount "+ temp_mount_point;
		if(result) {
			result = (boolean) sshClient.execjschForLogging(node_info, command, row_key, label).get("status");
		}
		// 8. mount file system
		if(result) {
			result = mountDisk(disk_info,node_info,lustre_fs,row_key, label);
		}
		
		
		return result;
	}


	/**
	 * mount disk
	 * @param disk_info
	 * @param node_info
	 * @param lustre_fs 
	 * @param row_key
	 * @param label
	 * @return
	 */
	private Boolean mountDisk(DiskInforVO disk_info, LustreNodesVO node_info, LustreFileSystemListVO lustre_fs, String row_key, String label) {
		String file_system_name = lustre_fs.getFs_name();
		String mount_point = "/mnt/"+file_system_name+"/" + disk_info.getDisk_type().toLowerCase() + disk_info.getIndex(); 
		String command = "mount -t lustre " + disk_info.getDisk_name() + " " + mount_point;
		Map<String, Object> tmp_map  = sshClient.execjschForLogging(node_info, command, row_key, label);
		return (Boolean) tmp_map.get("status");
	}



	/**
	 * umount disk
	 * @param disk_info
	 * @param node_info
	 * @param lustre_fs 
	 * @param row_key
	 */
	public Boolean umountDisk(DiskInforVO disk_info, LustreNodesVO node_info, LustreFileSystemListVO lustre_fs, String row_key, String Label) {
		// /mnt/ost6
		String file_system_name = lustre_fs.getFs_name();
		String mount_point = "/mnt/" + file_system_name+"/" +disk_info.getDisk_type().toLowerCase() + disk_info.getIndex(); 
		String command = "umount " + mount_point;
		Map<String, Object> tmp_map  = sshClient.execjschForLogging(node_info, command, row_key, Label);
		return (Boolean) tmp_map.get("status");
	}



	/**
	 * 각 노드들의 백업파일들을 읽어오는 메서드
	 * @param file_location2 
	 * @param string 
	 * @return
	 */
	public List<Map<String, Object>> findBackupfiles(Long file_system_num, String file_location) {
		if(file_location == null || "".equals(file_location)) {
			file_location = "/tmp";
		}
		
		// temp command
		String command = "";
		
		List<Map<String, Object>> result = new ArrayList<>();
		Map<String, Object> tmp_map = null;
		
		// 190401 je.kim view file_system
		LustreFileSystemListVO lfs_info =  new LustreFileSystemListVO();
		lfs_info.setNum(file_system_num);
		lfs_info = lustreDAO.viewFileSystem(lfs_info);
		
		// 190401 je.kim lustre file system name folder
		file_location += "/" +  lfs_info.getFs_name();
		
		
		LustreNodesVO find_node = new LustreNodesVO();
		find_node.setFile_system_num(file_system_num);
		List<LustreNodesVO> node_list =  getLustreNodes(find_node); // 전체리스트 가져오기
		
		for (LustreNodesVO nodesVO : node_list) {
			// 190403 je.kim create lustre backup location
			command = "mkdir -p " + file_location;
			sshClient.execjsch(nodesVO, command).get("log");
			
			// MDS 인지 OSS 인지 확인
			if(nodesVO.getNode_type().equals("MDS")) {
				command = "find "+file_location+" -name 'mds*.bak.tgz'";
			}else if(nodesVO.getNode_type().equals("OSS")) {
				command = "find "+file_location+" -name 'oss*.bak.tgz'";
			}else {
				// MDS OSS 을 제외한 나머지는 생략
				continue;
			}
			
			tmp_map = new HashMap<>();
			tmp_map.put("host_name", nodesVO.getHost_name());
			String ssh_result = (String) sshClient.execjsch(nodesVO, command).get("log");
			if(!ssh_result.equals("")) {
				tmp_map.put("backup_list",ssh_result.split("\n"));
//				tmp_map.put("device_list",getDiskDeviceList(nodesVO));
				tmp_map.put("device_list",new_getDiskDeviceList(nodesVO));
				tmp_map.put("node_info",nodesVO);
				result.add(tmp_map);
			}
			
		}
		
		return result;
	}



	/**
	 * read file system list
	 * @return
	 */
	public List<LustreFileSystemListVO> getFsList() {
		return lustreDAO.getFsList();
	}



	/**
	 * add file system
	 * @param file_system
	 * @return
	 */
	public Boolean addFileSystem(LustreFileSystemListVO file_system) {
		return lustreDAO.addFileSystem(file_system);
	}



	/**
	 * view file system
	 * @param file_system
	 * @return
	 */
	public LustreFileSystemListVO viewFileSystem(LustreFileSystemListVO file_system) {
		return lustreDAO.viewFileSystem(file_system);
	}



	/**
	 * modify file system
	 * @param file_system
	 * @return
	 */
	public Boolean setFileSystem(LustreFileSystemListVO file_system) {
		LustreFileSystemListVO temp_fs = viewFileSystem(file_system);
		// 파일시스템 정보가 없을경우 새로 생성
		if(temp_fs != null) {
			file_system.setOrigin_fs_name(temp_fs.getOrigin_fs_name());
			return lustreDAO.setFileSystem(file_system);
		}else {
			// 181227 je.kim 미리 파일시스템을 검색하고 없을경우 새로 작성 및 암바리를 통하여 디비에 적재
			return lustreDAO.addFileSystem(file_system) && lustreDAO.createLustreNodeForAabari(file_system);
		}
		
//		if(file_system.getNum() != null) {
//			if(lustreDAO.checkFileSystem(file_system) == 1)
//				return false;
//			else
//				return lustreDAO.setFileSystem(file_system);
//		}else {
//			if(lustreDAO.checkFileSystem(file_system) == 1)
//				return false;
//			else
//				return lustreDAO.addFileSystem(file_system);
//		}
	}
	
	
	/**
	 * @param file_system
	 * @param disk_info
	 * @param row_key 
	 * @return
	 */
	public Boolean add_filesystem_and_add_MGTDisk(LustreFileSystemListVO file_system, DiskInforVO disk_info, String row_key) {
		boolean result = false;
		
		
		LustreFileSystemListVO temp_fs = viewFileSystem(file_system);
		// 파일시스템 정보가 있는경우 false 처리
		if(temp_fs != null) {
			return false;
		}
		
		// 암바리에서 노드정보 가져오기
		List<LustreNodesVO> node_list = findLustreNodesForAmbari();
		Collection<LustreNodesVO> mds_node_list = CollectionUtils.select(node_list, new Predicate<Object>() {
			public boolean evaluate(Object a) {
				return ((LustreNodesVO) a).getNode_type().equals("MDS");
			}
		});
		if(mds_node_list.isEmpty()) {
			return null;
		}
		LustreNodesVO mds_node = mds_node_list.iterator().next();
		mds_node.setSsh_port(22);
		mds_node.setUser_id("root");
		mds_node.setPrivate_key(".ssh/id_rsa");
		
		// shkim 20190108 : MGT 추가는 MDS Setting 페이지에서 Network정보를 이용하여 LNET 추가 후에 동작 하는 것으로 수
		//result = addMGTDisk(file_system,mds_node,disk_info,ADD_MGT_LABEL,row_key);
		
		result = lustreDAO.addFileSystem(file_system);
		result = lustreDAO.createLustreNodeForAabari(file_system);
		LustreFileSystemListVO file_system_info = lustreDAO.viewFileSystem(file_system);
		disk_info.setIndex(0);
		disk_info.setFile_system_num(file_system_info.getNum());
		disk_info.setDisk_type("MDS");
		// shkim 20190108 lustre_nodes 테이블에서 MDS num 을 가져옴.
		disk_info.setLustre_nodes_key(lustreDAO.getLustreTypeNum(disk_info));
		disk_info.setDisk_type("MGT");
		save_disk_info(disk_info);
		return result;
	}
		
	/**
	 * @param file_system filesystem name
	 * @param mds_node node info
	 * @param disk_info disk info
	 * @param addMgtLabel label name
	 * @param row_key row key
	 * @return
	 */ 
	public boolean addMGTDisk(
			LustreFileSystemListVO file_system, 
			LustreNodesVO mds_node, 
			DiskInforVO disk_info, 
			String addMgtLabel, 
			String row_key) {
		boolean result = false;
		// LustreNodesVO 입력여부 검사
		if(
				mds_node.getNode_type() == null
				|| "".equals(mds_node.getNode_type())
		) {
			LOGGER.error("not Input LustreNodes Information");
			return false;
		}
		
		result = mgt_disk_format(mds_node,disk_info,row_key,addMgtLabel);
		result = mgt_disk_mount(mds_node,disk_info,file_system,row_key,addMgtLabel);
		
		return result;
	}



	/**
	 * MGT 디스크 마운트
	 * @param mds_node
	 * @param disk_info
	 * @param file_system
	 * @param row_key
	 * @param addMgtLabel
	 * @return
	 */
	public boolean mgt_disk_mount(LustreNodesVO mds_node, DiskInforVO disk_info, LustreFileSystemListVO file_system,
			String row_key, String addMgtLabel) {
		
		
		String device_name = disk_info.getDisk_name();
		// 디렉토리 생성 명령어
		// shkim 20190116 MGT 디렉토리 변경 /mnt/mgt0
		// String mgt_directory_name =  "/mnt/"+file_system.getFs_name()+"/"+"mgt0";
		String mgt_directory_name =  "/mnt/mgt0";
		String create_directory_command = "mkdir -p "+mgt_directory_name;
		sshClient.execjschForLogging(mds_node, create_directory_command, row_key , addMgtLabel);
		
		
		// 마운트 명령어
		String mgt_mount_command = "mount -t lustre "+device_name+" "+mgt_directory_name;
		Map<String, Object> result = sshClient.execjschForLogging(mds_node, mgt_mount_command, row_key , addMgtLabel);
				
		return (boolean) result.get("status");
	}



	/**
	 * MGS 디스크 포맷
	 * @param mds_node
	 * @param disk_info
	 * @param row_key
	 * @param addMgtLabel
	 * @return
	 */
	public boolean mgt_disk_format(LustreNodesVO mds_node, DiskInforVO disk_info, String row_key, String addMgtLabel) {
		String mgs_target_disk = disk_info.getDisk_name();
		String mgt_format_command = "mkfs.lustre --mgs --reformat " + mgs_target_disk ;
		
		Map<String, Object> result = sshClient.execjschForLogging(mds_node, mgt_format_command, row_key , addMgtLabel);
		
		return (boolean) result.get("status");
	}



	/**
	 * mdt 추가
	 * @param lustreNodesVO ssh 명령어 전송할 호스트정보
	 * @param diskInforVO 디비에 저장시킬 디스크 정보
	 * @param lustre_fs 러스터 파일시스템 정보
	 * @param row_key 로그에 저장시킬 랜덤키값
	 */
	public Boolean mdsAddDisk(
			LustreNodesVO lustreNodesVO, 
			DiskInforVO diskInforVO, 
			LustreFileSystemListVO lustre_fs, 
			String row_key
			){
		
		// 디비에서 가져오기
		lustre_fs = viewFileSystem(lustre_fs);
		if(lustre_fs == null) {
			return false;
		}
		
		// LustreNodesVO 입력여부 검사
		if(
				lustreNodesVO.getNode_type() == null
				|| "".equals(lustreNodesVO.getNode_type())
				|| lustreNodesVO.getIndex() == null
		) {
			LOGGER.error("not Input LustreNodes Information");
			return false;
		}
		// LustreNodes 가져오기
		
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			LOGGER.error("not find LustreNodes Information");
			return false;
		}
		LustreNodesVO lustre_info = lustreNodes.get(0);
		
		
		// 러스터 커널 설정(네트워크 설정)
		// 19.01.02 je.kim setNetwork_device 한번이라도 동작했으면 스킵처리
		if(isSetLustreConf()) {
			// 만약에 네트워크 명이 있으면 업데이트
			if(
					lustreNodesVO.getNetwork_device() != null && 
					!"".equals(lustreNodesVO.getNetwork_device())
					) {
				lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
				lustreDAO.setLustreNodes(lustre_info);
			}
			setLustreConf(lustre_info,row_key, "LNET Setting");
		// 한번이상 동작시 네트워크 설정만 업데이트 처리
		}else {
			String log_massege = "Lustre settings have been detected at least once.\n Skip Lustre Settings...";
			lustreDAO.saveCommandLog(row_key, log_massege, "LNET Setting", "info", lustre_info.getHost_name());
			lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
			lustreDAO.setLustreNodes(lustre_info);
		}
		
		// 해당 노드가 MDS 인지 OSS 인지 판별
		if(lustre_info.getNode_type().equals("MDS")) {
			if(lustreDAO.checkMGT() == 1){
				// shkim 20190109 MGT 디바이스 추가 로직 추가
				DiskInforVO mgt_disk_info = new DiskInforVO();
				// disk_name 가져오는 부분 추가. lustre_node_key, disk_type(MGT)로 disk_name를 가져옴.
				mgt_disk_info.setLustre_nodes_key(lustre_info.getNum());
				mgt_disk_info.setDisk_type("MGT");			
				mgt_disk_info.setDisk_name(lustreDAO.getMGTDisk_Name(mgt_disk_info));
				
				addMGTDisk(lustre_fs, lustre_info, mgt_disk_info, ADD_MGT_LABEL, IdGeneration.generateId(16));
			}			
			return addMDTDisk(lustre_info,diskInforVO , lustre_fs ,IdGeneration.generateId(16), MDS_SETTING_LABEL);
		}else {
			LOGGER.error("not MDS lustre disk type");
			return false;
		}
		
	}
	
	/**
	 * setNetwork_devic 메서드 실행여부 검사 
	 * (file system 테이블에서 step.2 이상 되는 경우 false, 없으면 true)
	 * @return
	 */
	public Boolean isSetLustreConf() {
		return lustreDAO.isSetLustreConf();
	}
	
	/**
	 * ost 추가
	 * @param lustreNodesVO
	 * @param lustre_fs 
	 * @param row_key
	 */
	public Boolean ossAddDisk(LustreNodesVO lustreNodesVO, LustreFileSystemListVO lustre_fs, String row_key) {
		// file system 검사
		if(lustre_fs == null) {
    		return false;
    	}
		
		// LustreNodesVO 입력여부 검사
		boolean result = false;
		if(
			lustreNodesVO.getNode_type() == null
			|| "".equals(lustreNodesVO.getNode_type())
			|| lustreNodesVO.getIndex() == null
		) {
			return false;
		}
		// LustreNodes 가져오기
		
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		
		
		List<DiskInforVO> disk_list = lustreNodesVO.getDisk_list();
		
		LustreNodesVO lustre_info = lustreNodes.get(0);
		

		// 러스터 커널 설정(네트워크 설정)
		// 19.01.02 je.kim setNetwork_device 한번이라도 동작했으면 스킵처리
		//if(isSetLustreConf()) {
			// 만약에 네트워크 명이 있으면 업데이트
			if(lustreNodesVO.getNetwork_device() != null && !"".equals(lustreNodesVO.getNetwork_device())) {
				lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
				lustreDAO.setLustreNodes(lustre_info);
			}
			
			// 러스터 커널 설정(네트워크 설정)
			setLustreConf(lustre_info,row_key, OSS_SETTING_LABEL);
//		}else {
//			String log_massege = "Lustre settings have been detected at least once.\n Skip Lustre Settings...";
//			lustreDAO.saveCommandLog(row_key, log_massege, OSS_SETTING_LABEL, "info", lustre_info.getHost_name());
//			lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
//			lustreDAO.setLustreNodes(lustre_info);
//		}
		
//		// 삭제용 임시 row_key 생성
//		String remove_ost_row_key = IdGeneration.generateId(16);
//		// 저장된 디스크 리스트를 돌면서 remove_disk_frag 가 true 일경우 삭제 동작 실시
//		for (DiskInforVO disk : disk_list) {
//			if(disk.getRemove_disk_frag() != null && disk.getRemove_disk_frag()) {
//				lustreService.removeOSTDisk(lustre_info, disk,remove_ost_row_key,OST_REMOVING_LABEL);
//			}
//		}
		
		
		if(lustre_info.getNode_type().equals("OSS")) {
			result = addOSTDisk(lustre_info,disk_list,lustre_fs,row_key, OSS_SETTING_LABEL);
		}
		return result;
	}
	
	
	
	/**
	 * update client folder
	 * @param lustreNode
	 * @param fs_data
	 * @return
	 */
	public Boolean clientMountFolderSetting(LustreNodesVO lustreNode, LustreFileSystemListVO fs_data) {
		LustreNodesVO search_node = new LustreNodesVO();
		boolean result = false;
		search_node.setNode_type("CLIENT");
		search_node.setFile_system_num(fs_data.getNum());
		List<LustreNodesVO> client_node_list = getLustreNodes(search_node);
		
		LOGGER.info("clientMountFolderSetting => umount {}",fs_data.getFs_name());
		result = umountClientFolder(fs_data,IdGeneration.generateId(16));
		for (LustreNodesVO client_node : client_node_list) {
			client_node.setLustre_client_folder(lustreNode.getLustre_client_folder());
			// 디비 업데이트
			result = lustreDAO.setLustreNodes(client_node);
		}
		LOGGER.info("clientMountFolderSetting => mount {}",fs_data.getFs_name());
		result = mountClientFolder(fs_data,IdGeneration.generateId(16));
		return result;
	}
	
	/**
	 * Lustre client setting
	 * @param fs_data 
	 * @param lustreNodesVO
	 * @param row_key
	 */
	//@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public Boolean clientSetting(LustreNodesVO lustreNode, LustreFileSystemListVO fs_data, String row_key) {
		boolean result = false;
		
		// LustreNodesVO 입력여부 검사
		if(
				lustreNode.getNode_type() == null
			|| "".equals(lustreNode.getNode_type())
			|| lustreNode.getIndex() == null
		) {
			new Exception("not node type setting");
		}
		// 190410 je.kim set file system
		lustreNode.setFile_system_num(fs_data.getNum());
		
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNode);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			new Exception("not found node");
		}
		LustreNodesVO lustre_info = lustreNodes.get(0);

		
		// 만약에 네트워크 명이 있으면 업데이트
		if(lustreNode.getNetwork_device() != null || !"".equals(lustreNode.getNetwork_device())) {
			lustre_info.setNetwork_device(lustreNode.getNetwork_device());
			// 러스터 커널 설정(네트워크 설정)
			result = setLustreConf(lustre_info, row_key, CLIENT_SETTING_LABEL);
		}
		
		if(lustreNode.getLustre_client_folder() != null && !"".equals(lustreNode.getLustre_client_folder())) {
			lustre_info.setLustre_client_folder(lustreNode.getLustre_client_folder());
		}
		
		// 디비 업데이트
		lustreDAO.setLustreNodes(lustre_info);
		
		if(result) {
			fs_data.setFs_step(ADD_CLIENT_STEP);
			fs_data.setOrigin_fs_name(fs_data.getFs_name());
			setFileSystem(fs_data);
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
		return lustreDAO.createLustreNodeForAabari(file_system);
	}



	/**
	 * 설치된 MGT 디스크 정보 가져오기
	 * @return
	 */
	public DiskInforVO get_MGT_Disk() {
		DiskInforVO search_disk = new DiskInforVO();
		search_disk.setDisk_type("MGT");
		List<DiskInforVO> disk_list = lustreDAO.getDisks(search_disk);
		return (disk_list.size() > 0) ? disk_list.get(0) : null;
	}
	
	
	/**
	 * OST 제거
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @param file_system 
	 * @param radom_key
	 */
	public Boolean no_async_removeOST(LustreNodesVO lustreNodesVO, DiskInforVO diskInforVO, LustreFileSystemListVO file_system, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
				|| "".equals(lustreNodesVO.getHost_name())
				|| diskInforVO.getDisk_name() == null
				|| "".equals(diskInforVO.getDisk_name())
			) {
				return false;
		}
			
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO ostInfo = lustreNodes.get(0);
		
		diskInforVO.setLustre_nodes_key(ostInfo.getNum());
		
		List<DiskInforVO> disk_list = getDisks(diskInforVO);
		if(disk_list.size() <= 0) {
			return false;
		}
				
		DiskInforVO diskInfo = disk_list.get(0);
		
		return removeOSTDisk(ostInfo, diskInfo ,file_system,row_key);
	}

	
	
	
	/**
	 * mds 호스트 네임을 통해서 아이피를 추출하는 메서드 
	 * @param networkDeviceName
	 * @return ipaddress
	 */
	public String get_MDS_IP_Address(String networkDeviceName) {
		
		// DB 에서 MDS 노드정보 가져오기
		LustreNodesVO mds_node = new LustreNodesVO();
		mds_node.setNode_type("MDS");
		List<LustreNodesVO> mds_node_list = lustreDAO.getLustreNodes(mds_node);
		if(mds_node_list.size() > 0) {
			// 만약 DB 에 MDS 노드가 존재하면
			mds_node = mds_node_list.get(0);
		}else {
			mds_node = getAmbariForMDSnode();
		}
		
		String command = "ifconfig "+networkDeviceName+" | grep -w 'inet' | awk '{print $2}' ";
		// ssh 결과
		String result = ((String) sshClient.execjsch(mds_node, command).get("log")).replaceAll("\n", "");
		LOGGER.info("get_MDS_IP_Address : {}",result);
		if(result.contains("error") || result.contains("command not found") || result.contains("Device not found")) {
			return null;
		}else {
			return result;
		}
	}



	/**
	 * remove file system
	 * @param file_system
	 * @return
	 */
	public Map<String, Object> removeLustreFilesystem(LustreFileSystemListVO file_system) {
		Map<String, Object> result = new HashMap<>();
		file_system = lustreDAO.viewFileSystem(file_system);
		boolean check_logic = delete_lustre_file_system(file_system);
		if(check_logic) {
			result.put("result",lustreDAO.removeLustreFilesystem(file_system));
		}
		//result.put("result",true);
		return result;
	}



	/**
	 * delete lustre file system
	 * @param file_system
	 * @return
	 */
	private Boolean delete_lustre_file_system(LustreFileSystemListVO file_system) {
		
		boolean result = false;
		
		// get oss nodes
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setNode_type("OSS");
		search_node.setFile_system_num(file_system.getNum());
		List<LustreNodesVO> OSS_nodes = lustreDAO.getLustreNodes(search_node);
		
		// get mds node
		search_node.setNode_type("MDS");
		search_node.setFile_system_num(file_system.getNum());
		List<LustreNodesVO> MDS_nodes = lustreDAO.getLustreNodes(search_node);
		LustreNodesVO MDS_node = null;
		if(MDS_nodes.size() > 0) {
			MDS_node = MDS_nodes.get(0);
		}
		
		// get mdt disk
		DiskInforVO mdt_disk = null;
		List<DiskInforVO> MDS_disk_list = lustreDAO.getDiskInfo(MDS_node.getNum());
		if(MDS_disk_list.size() > 0) {
			mdt_disk = MDS_disk_list.get(0);
		}
		
		// command
		String command = "";
		
		// step 1. Unmount all clients using this file system
		result = umountClientFolder(file_system,IdGeneration.generateId(16));
		
		// step 2. remove OSTs file system
		for (LustreNodesVO oss_node : OSS_nodes) {
			String OSS_umount_rowkey = IdGeneration.generateId(16);
			String ost_remove_title = "";
			
			// get disk list
			List<DiskInforVO> oss_disk = oss_node.getDisk_list();
			// remove all ost disk -> umount ost disks
			for (DiskInforVO OST_disk_info : oss_disk) {
				// migration 간에 시간이 소요됨으로 umount 처리
				// removeOSTDisk(oss_node, OST_disk_info, file_system, IdGeneration.generateId(16));
				
				String OST_mount_folder = "/mnt/" + file_system.getFs_name() + "/ost" + OST_disk_info.getIndex();
				
				if(checkMountDisk(oss_node , OST_mount_folder)) {
					ost_remove_title = "umount " + OST_disk_info.getDisk_name();
					
					command = "fuser -ck "+OST_mount_folder;
					result = (boolean)sshClient.execjschForLogging(oss_node, command, OSS_umount_rowkey, ost_remove_title).get("status");
					
					command = "umount -l " + OST_mount_folder;
					result = (boolean)sshClient.execjschForLogging(oss_node, command, OSS_umount_rowkey, ost_remove_title).get("status");
				}
			}
		}
		
		//  step 3. remove MDT
		String MDT_mount_folder = "/mnt/" + file_system.getFs_name() + "/mdt0";
		if(checkMountDisk(MDS_node , MDT_mount_folder)) {
			String mdt_remove_title = "umount " + mdt_disk.getDisk_name();
			String MDS_umount_rowkey = IdGeneration.generateId(16);
			command = "fuser -ck "+MDT_mount_folder;
			result = (boolean)sshClient.execjschForLogging(MDS_node, command, MDS_umount_rowkey, mdt_remove_title).get("status");
			
			command = "umount -l " + MDT_mount_folder;
			result = (boolean)sshClient.execjschForLogging(MDS_node, command, MDS_umount_rowkey, mdt_remove_title).get("status");
		}
		
		
		return result;
	}



	/**
	 * network stop (no async)
	 * @param lustreNodesVO
	 * @param row_key
	 * @return
	 */
	public Boolean no_async_networkStop(LustreNodesVO lustreNodesVO, String row_key) {
		if(
			lustreNodesVO.getHost_name() == null
			|| "".equals(lustreNodesVO.getHost_name())
		) {
			return false;
		}
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		String label = lustreNodesVO.getHost_name() + " LNET Network Stop";
		LustreNodesVO lustreNode = lustreNodes.get(0);
		LustreFileSystemListVO lustre_fs = new LustreFileSystemListVO();
		lustre_fs.setNum(lustreNodesVO.getFile_system_num());
		lustre_fs = viewFileSystem(lustre_fs);		
		return new_networkStop(lustreNode,lustre_fs,label,row_key);
	}



	/**
	 * network start  (no async)
	 * @param lustreNodesVO
	 * @param Lustrecontent
	 * @param row_key
	 * @return
	 */
	public Boolean no_async_networkStart(LustreNodesVO lustreNodesVO, String Lustrecontent, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
			|| "".equals(lustreNodesVO.getHost_name())
		) {
			return false;
		}
		List<LustreNodesVO> lustreNodes = getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return false;
		}
		LustreNodesVO lustreNode = lustreNodes.get(0);
		String label = lustreNodesVO.getHost_name() + " LNET Network Start";
		
		LustreFileSystemListVO lustre_fs = new LustreFileSystemListVO();
		lustre_fs.setNum(lustreNodesVO.getFile_system_num());
		lustre_fs = viewFileSystem(lustre_fs);
		
		return networkStart(lustreNode,Lustrecontent,lustre_fs,label,row_key);
	}

	
}

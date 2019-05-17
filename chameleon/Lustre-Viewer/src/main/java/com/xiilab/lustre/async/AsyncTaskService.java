package com.xiilab.lustre.async;

import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import com.xiilab.lustre.api.LustreDAO;
import com.xiilab.lustre.api.LustreService;
import com.xiilab.lustre.model.DiskInforVO;
import com.xiilab.lustre.model.LustreFileSystemListVO;
import com.xiilab.lustre.model.LustreNodesVO;
import com.xiilab.lustre.utilities.IdGeneration;
import com.xiilab.lustre.utilities.SshClient;

@Service("asyncTaskService")
public class AsyncTaskService {
	
	private final static String MDS_SETTING_LABEL = "MDS Setting";
	private final static String OSS_SETTING_LABEL = "OSS Setting";
//	private final static String OST_REMOVING_LABEL = " Removing";
	private final static String CLIENT_SETTING_LABEL = "Client Setting";
	
	@Autowired
	private LustreService lustreService;
	
	@Autowired
	private SshClient sshClient;
	
	@Autowired 
	private LustreDAO lustreDAO;
	
	static final Logger LOGGER = LoggerFactory.getLogger(AsyncTaskService.class);
	
	/**
	 * 디스크 백업
	 * @param disk
	 * @param lustre_fs 
	 * @param row_key
	 * @return
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public Boolean backupDisk(DiskInforVO disk, LustreFileSystemListVO lustre_fs, String row_key) {
		// 저장된 디스크 정보 가져오기
		List<DiskInforVO> disk_list = lustreService.getDisks(disk);
		
		// 저장된 노드 정보 가져오기
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setHost_name(disk.getHost_name());
		List<LustreNodesVO> node_list = lustreService.getLustreNodes(search_node);
		
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

		
		return lustreService.backupDisk(disk_info,node_info,lustre_fs,row_key,Label);
	}
	
	
	
	/**
	 * 디스크 복구
	 * @param disk
	 * @param lustre_fs 
	 * @param radom_key
	 * @return
	 */
	public Boolean restoreDisk(DiskInforVO disk, LustreFileSystemListVO lustre_fs, String row_key) {
		
		// 저장된 노드 정보 가져오기
		LustreNodesVO search_node = new LustreNodesVO();
		search_node.setHost_name(disk.getHost_name());
		List<LustreNodesVO> node_list = lustreService.getLustreNodes(search_node);
		
		// 디비에서 가져오기
    	lustre_fs = lustreService.viewFileSystem(lustre_fs);
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
		
		// 181226 je.kim 디비에 저장된 디스크 정보가 없을경우 디비에 추가하는 로직 추가
		// 저장된 디스크 정보 가져오기
//		DiskInforVO search_disk = new DiskInforVO();
//		search_disk.setDisk_name(disk.getDisk_name());
//		search_disk.setLustre_nodes_key(node_info.getNum());
//		List<DiskInforVO> disk_list = lustreService.getDisks(search_disk);
//		
//		// 만약 검색된 데이터가 없을경우 예외처리
//		if(disk_list.size() == 0) {
//			LOGGER.error("not found disk information");
//			return false;
//			//new Exception("not found disk information");
//		}
		
//		DiskInforVO disk_info = disk_list.get(0);
		// 백업된 파일위치
//		disk_info.setBackup_file_localtion(disk.getBackup_file_localtion());
		// 새로운 디바이스 명으로 교체
//		disk_info.setDisk_name(disk.getDisk_name());
		
		disk.setLustre_nodes_key(node_info.getNum());
		
		// 라벨정의
		String Label = "Restore "+disk.getDisk_type()+disk.getIndex()+" For " + disk.getDisk_name();
		
		
		return lustreService.restoreDisk(disk,node_info,lustre_fs,row_key,Label);
	}
	
	/**
	 * mdt 추가
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @param lustre_fs 러스터 파일시스템 정보
	 * @param row_key
	 */
	@Async("executorLustreManager")
	public void mdsAddDisk(LustreNodesVO lustreNodesVO, DiskInforVO diskInforVO,LustreFileSystemListVO lustre_fs,  String row_key){
		
		// 디비에서 가져오기
		lustre_fs = lustreService.viewFileSystem(lustre_fs);
		if(lustre_fs == null) {
			return;
		}
		
		// LustreNodesVO 입력여부 검사
		if(
				lustreNodesVO.getNode_type() == null
				|| "".equals(lustreNodesVO.getNode_type())
				|| lustreNodesVO.getIndex() == null
		) {
			return;
		}
		
		
		// LustreNodes 가져오기
		
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
		LustreNodesVO lustre_info = lustreNodes.get(0);
		// 만약에 네트워크 명이 있으면 업데이트
		if(lustreNodesVO.getNetwork_device() != null && !"".equals(lustreNodesVO.getNetwork_device())) {
			lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
			lustreDAO.setLustreNodes(lustre_info);
		}
		
		// 러스터 커널 설정(네트워크 설정)
		lustreService.setLustreConf(lustre_info,row_key, MDS_SETTING_LABEL);
		
		// 해당 노드가 MDS 인지 OSS 인지 판별
		if(lustre_info.getNode_type().equals("MDS")) {
			lustreService.addMDTDisk(lustre_info,diskInforVO,lustre_fs,row_key, MDS_SETTING_LABEL);
		}
		
	}
	
	
	/**
	 * ost 추가
	 * @param lustreNodesVO
	 * @param row_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void ossAddDisk(LustreNodesVO lustreNodesVO,LustreFileSystemListVO lustre_fs, String row_key) {
		// 디비에서 가져오기
		lustre_fs = lustreService.viewFileSystem(lustre_fs);
		if(lustre_fs == null) {
			return;
		}
		
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getNode_type() == null
			|| "".equals(lustreNodesVO.getNode_type())
			|| lustreNodesVO.getIndex() == null
		) {
			return;
		}
		// LustreNodes 가져오기
		
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
		
		List<DiskInforVO> disk_list = lustreNodesVO.getDisk_list();
		
		LustreNodesVO lustre_info = lustreNodes.get(0);
		// 만약에 네트워크 명이 있으면 업데이트
		if(lustreNodesVO.getNetwork_device() != null && !"".equals(lustreNodesVO.getNetwork_device())) {
			lustre_info.setNetwork_device(lustreNodesVO.getNetwork_device());
			lustreDAO.setLustreNodes(lustre_info);
		}
		
		// 러스터 커널 설정(네트워크 설정)
		lustreService.setLustreConf(lustre_info,row_key, OSS_SETTING_LABEL);
		
//		// 삭제용 임시 row_key 생성
//		String remove_ost_row_key = IdGeneration.generateId(16);
//		// 저장된 디스크 리스트를 돌면서 remove_disk_frag 가 true 일경우 삭제 동작 실시
//		for (DiskInforVO disk : disk_list) {
//			if(disk.getRemove_disk_frag() != null && disk.getRemove_disk_frag()) {
//				lustreService.removeOSTDisk(lustre_info, disk,remove_ost_row_key,OST_REMOVING_LABEL);
//			}
//		}
		
		
		if(lustre_info.getNode_type().equals("OSS")) {
			lustreService.addOSTDisk(lustre_info,disk_list,lustre_fs,row_key, OSS_SETTING_LABEL);
		}
		
	}
	
	/**
	 * OST 제거
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @param file_system 
	 * @param radom_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void removeOST(LustreNodesVO lustreNodesVO, DiskInforVO diskInforVO, LustreFileSystemListVO file_system, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
				|| "".equals(lustreNodesVO.getHost_name())
				|| diskInforVO.getDisk_name() == null
				|| "".equals(diskInforVO.getDisk_name())
			) {
				return;
		}
			
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
		LustreNodesVO ostInfo = lustreNodes.get(0);
		
		diskInforVO.setLustre_nodes_key(ostInfo.getNum());
		
		List<DiskInforVO> disk_list = lustreService.getDisks(diskInforVO);
		if(disk_list.size() <= 0) {
			return;
		}
				
		DiskInforVO diskInfo = disk_list.get(0);
		
		lustreService.removeOSTDisk(ostInfo, diskInfo ,file_system,row_key);
	}
	
	
	/**
	 * Lustre client setting
	 * @param lustreNodesVO
	 * @param row_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void clientSetting(LustreNodesVO lustreNode, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
				lustreNode.getNode_type() == null
			|| "".equals(lustreNode.getNode_type())
			|| lustreNode.getIndex() == null
		) {
			new Exception("not node type setting");
		}
		
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNode);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			new Exception("not found node");
		}
		LustreNodesVO lustre_info = lustreNodes.get(0);
		// 만약에 네트워크 명이 있으면 업데이트
		if(lustreNode.getNetwork_device() != null || !"".equals(lustreNode.getNetwork_device())) {
					
			if(lustreNode.getNetwork_device() != null && lustreNode.getLustre_client_folder() != null) {
				lustre_info.setNetwork_device(lustreNode.getNetwork_device());
			}
					
			if(lustreNode.getLustre_client_folder() != null && !"".equals(lustreNode.getLustre_client_folder())) {
				lustre_info.setLustre_client_folder(lustreNode.getLustre_client_folder());
			}
					
			lustreDAO.setLustreNodes(lustre_info);
		}
				
		// 러스터 커널 설정(네트워크 설정)
		lustreService.setLustreConf(lustre_info, row_key, CLIENT_SETTING_LABEL);
	}

	
	/**
	 * Lustre oss activate
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @param file_system 
	 * @param row_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void activate(LustreNodesVO lustreNodesVO, DiskInforVO diskInforVO, LustreFileSystemListVO file_system, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
			|| "".equals(lustreNodesVO.getHost_name())
			|| diskInforVO.getDisk_name() == null
			|| "".equals(diskInforVO.getDisk_name())
		) {
			return;
		}
		
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
		LustreNodesVO ostInfo = lustreNodes.get(0);
		
		diskInforVO.setLustre_nodes_key(ostInfo.getNum());
		
		List<DiskInforVO> disk_list = lustreService.getDisks(diskInforVO);
		if(disk_list.size() <= 0) {
			return;
		}
		
		DiskInforVO diskInfo = disk_list.get(0);
		String ossSettingLabel = "OST"+(diskInfo.getIndex())+" Activate";
		
		if(lustreService.activate(ostInfo,diskInfo,file_system,row_key,ossSettingLabel)) {
			diskInfo.setIs_activate(true);
			lustreService.updateDisk(diskInfo);
		}
	}
	
	
	/**
	 * Lustre oss deactivate
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @param file_system 
	 * @param row_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void deactivate(LustreNodesVO lustreNodesVO, DiskInforVO diskInforVO, LustreFileSystemListVO file_system, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
			|| "".equals(lustreNodesVO.getHost_name())
			|| diskInforVO.getDisk_name() == null
			|| "".equals(diskInforVO.getDisk_name())
		) {
			return;
		}
		
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
		LustreNodesVO ostInfo = lustreNodes.get(0);
		
		diskInforVO.setLustre_nodes_key(ostInfo.getNum());
		
		List<DiskInforVO> disk_list = lustreService.getDisks(diskInforVO);
		if(disk_list.size() <= 0) {
			return;
		}
		
		DiskInforVO diskInfo = disk_list.get(0);
		String ossSettingLabel = "OST"+(diskInfo.getIndex())+" Deactivate";
		
		if(lustreService.deactivate(ostInfo,diskInfo,file_system,row_key,ossSettingLabel)) {
			diskInfo.setIs_activate(false);
			lustreService.updateDisk(diskInfo);
		}
	}
	
	/**
	 * LNET Setting Network Start
	 * @param lustreNodesVO
	 * @param Lustrecontent
	 * @param row_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void networkStart(LustreNodesVO lustreNodesVO, String Lustrecontent, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
			|| "".equals(lustreNodesVO.getHost_name())
		) {
			return;
		}
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
		LustreNodesVO lustreNode = lustreNodes.get(0);
		String label = lustreNodesVO.getHost_name() + " LNET Network Start";
		
		LustreFileSystemListVO lustre_fs = new LustreFileSystemListVO();
		lustre_fs.setNum(lustreNodesVO.getFile_system_num());
		lustre_fs = lustreService.viewFileSystem(lustre_fs);
		
		lustreService.networkStart(lustreNode,Lustrecontent,lustre_fs,label,row_key);
	}
	
	/**
	 * LNET Setting Network Stop
	 * @param search_node
	 * @param radom_key
	 */
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
	public void networkStop(LustreNodesVO lustreNodesVO, String row_key) {
		// LustreNodesVO 입력여부 검사
		if(
			lustreNodesVO.getHost_name() == null
			|| "".equals(lustreNodesVO.getHost_name())
		) {
			return;
		}
		List<LustreNodesVO> lustreNodes = lustreService.getLustreNodes(lustreNodesVO);
		// 만약에 내용이 없으면 종료
		if(lustreNodes.size() <= 0) {
			return;
		}
//		LustreNodesVO lustreNode = lustreNodes.get(0);
//		
//		String label = lustreNodesVO.getHost_name() + " LNET Network Stop";
//		
//		// 해당호스트내의 모든 마운트 폴더들을 언마운트후에 네트워크 스탑처리
//		lustreService.new_networkStop(lustreNode,label,row_key);
		
		
		
		String label = lustreNodesVO.getHost_name() + " LNET Network Stop";
		LustreNodesVO lustreNode = lustreNodes.get(0);
		LustreFileSystemListVO lustre_fs = new LustreFileSystemListVO();
		lustre_fs.setNum(lustreNodesVO.getFile_system_num());
		lustre_fs = lustreService.viewFileSystem(lustre_fs);		
		lustreService.new_networkStop(lustreNode,lustre_fs,label,row_key);
		
	}
	
   
	@Async("executorLustreManager") // AsyncConfig 에 있는 executorLustreManager Bean 과 연동
    public void executorSample(String str) {
        // LOG : 시작 입력
        // ...
        System.out.println("==============>>>>>>>>>>>> THREAD START");
        
        // 내용
        // 내용
        // 내용
        
//        while (true) {
//			System.out.println("test!");
//			
//		}
        
        
        // LOG : 종료 입력
        // ...
        System.out.println("==============>>>>>>>>>>>> THREAD END");
    }

	

	
}

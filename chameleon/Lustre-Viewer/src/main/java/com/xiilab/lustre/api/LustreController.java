package com.xiilab.lustre.api;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.task.TaskRejectedException;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import com.xiilab.lustre.model.DiskInforVO;
import com.xiilab.lustre.model.LustreFileSystemListVO;
import com.xiilab.lustre.model.LustreLogVO;
import com.xiilab.lustre.model.LustreNodesVO;
import com.xiilab.lustre.utilities.AmbariProperty;
import com.xiilab.lustre.utilities.IdGeneration;

@Controller
public class LustreController {
	@Autowired
	private LustreService lustreService;
	
	@Autowired
	private AmbariProperty ambariProperty;
	
	
	
	/**
	 * 181227 je.kim
	 * 암바리 API을 통하여 러스터 정보들을 삽입하는 메서드 
	 * @param file_system
	 * @return
	 */
	@RequestMapping(value = "/api/v1/ambari/createLustreNodeForAabari", method = {RequestMethod.GET, RequestMethod.POST})
	public @ResponseBody Map<String, Object> createLustreNodeForAabari(LustreFileSystemListVO file_system) {
		Map<String, Object> result = new HashMap<>();
		result.put("result", lustreService.createLustreNodeForAabari(file_system));
		return result;
	}
	
	
	/**
	 * 디비에 적재되어 있는 러스터정보을 읽어오고 없으면 암바리를 통하여 동기화 하는 메서드
	 * @return
	 */
	@RequestMapping(value = "/api/v1/ambari/syncLustreTable", method = RequestMethod.GET)
	public @ResponseBody Map<String, Object> syncLustreTable(LustreFileSystemListVO file_system) {
		Map<String, Object> result = new HashMap<>();
		result.put("result", lustreService.syncLustreTable(file_system));
		return result;
	}
	
	/**
	 * 암바리 api을 통한 lustre node list 출력
	 * @return
	 */
	@RequestMapping(value = "/api/v1/ambari/findLustreNodesForAmbari", method = RequestMethod.GET)
	public @ResponseBody Map<String, Object> findLustreNodesForAmbari() {
		Map<String, Object> result = new HashMap<>();
		result.put("result", lustreService.findLustreNodesForAmbari());
		return result;
	}
	
	/**
	 * Lustre 정보 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/ambari/getLustreNodes",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<LustreNodesVO> getLustreNodes(LustreNodesVO lustreNodesVO) {
		return lustreService.getLustreNodes(lustreNodesVO);
	}
	
		
	/**
	 * Ambari 컴포넌트 리스트 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/ambari/getNodeMatricDataArr",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody String getNodeDataArr() {
		return lustreService.getNodeMatricDataArr();
	}
	
	/**
	 * 네트워크 디바이스 정보 가져오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/getNetWorkDevice",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> getNetWorkDevice(LustreNodesVO lustreNodesVO){
		return lustreService.getNetWorkDevice(lustreNodesVO);
	}
	
	/**
	 *디스크 정보 가져오기 (fdisk -l)
	 * @param lustreNodesVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/getDiskDevice",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> getDiskDevice(LustreNodesVO lustreNodesVO){
		return lustreService.getDiskDevice(lustreNodesVO);
	}
	
	/**
	 *디스크 정보 가져오기 (lsblk -o NAME,FSTYPE,MOUNTPOINT -ds | awk '$3 == "" {print $1}')
	 * @param lustreNodesVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/new_getDiskDevice",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> new_getDiskDevice(LustreNodesVO lustreNodesVO){
		return lustreService.new_getDiskDevice(lustreNodesVO);
	}
	
	/**
	 * 디스크 추가하기
	 * @param lustreNodesVO
	 * @param diskInforVO
	 * @return
	 */
//	@RequestMapping(value="/api/v1/lustre/addDisk",method = {RequestMethod.GET,RequestMethod.POST})
//	public @ResponseBody Map<String, Object> addDIsk(LustreNodesVO lustreNodesVO,DiskInforVO diskInforVO){
//		return lustreService.addDIsk(lustreNodesVO,diskInforVO);
////		return null;
//	}
	
	
	/**
	 * 테이블 생성 테스트
	 * @return
	 */
	@RequestMapping(value = "/api/v1/lustre/chechCreatedTables", method = RequestMethod.GET)
	public @ResponseBody Map<String, Object> chechCreatedTables(){
		return lustreService.checkCreatedTables();
	}
	
	
	
	/**
	 * 로그 리스트 불려오기
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/getLogList",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<LustreLogVO> getLogList(){
		return lustreService.getLogList();
	}
	
	
	/**
	 * 로그 내용 보기
	 * @param lustreLogVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/viewLog",method = {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<LustreLogVO> viewLog(LustreLogVO lustreLogVO){
		return lustreService.viewLog(lustreLogVO);
	}
	
	
	
	/**
	 * 최근 로그 내용보기
	 * @param lustreLogVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/viewLastLogLine",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<LustreLogVO> viewLastLogLine(LustreLogVO lustreLogVO) {
		return lustreService.viewLastLogLine(lustreLogVO);
	}
	
	
	/**
	 * lustre 설치된 노드들의 lustre conf 파일을 읽어오는 메서드
	 * @param lustreNodesVO
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/readLustreConf",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<Map<String,Object>> readLustreConf(LustreNodesVO lustreNodesVO) {
		return lustreService.readLustreConf(lustreNodesVO);
	}
	
	/**
	 * lustre client 의 마운트 폴더를 읽어오는 매서드 
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/getClientFolder",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody String getClientFolder(
			@RequestParam(value="file_system_num",required=true) Long file_system_num
			) {
		return lustreService.getClientFolder(file_system_num);
	}
	
	/**
	 * 각 노드들의 백업파일들을 읽어오는 메서드
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/findBackupfiles",method= {RequestMethod.GET,RequestMethod.POST})
	public  @ResponseBody List<Map<String,Object>> findBackupfiles(
			@RequestParam(value="file_location",required=false) String file_location,
			@RequestParam(value="file_system_num",required=true) Long file_system_num
		){
		return lustreService.findBackupfiles(file_system_num,file_location);
	}
	
	/**
	 * read file system list
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/getFsList",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody List<LustreFileSystemListVO> getFsList(){
		return lustreService.getFileSystemIsMountList();
	}
	
	/**
	 * input FileSystem
	 * @param file_system
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/addFileSystem",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> addFileSystem(LustreFileSystemListVO file_system){
		Map<String, Object> result = new HashMap<>();
		result.put("status", lustreService.addFileSystem(file_system));
		return result;
	}
	
	/**
	 * view file system
	 * @param file_system
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/viewFileSystem",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody LustreFileSystemListVO viewFileSystem(LustreFileSystemListVO file_system){
		return lustreService.viewFileSystem(file_system);
	}
	
	/**
	 * view file system
	 * @param file_system
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/setFileSystem",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> setFileSystem(LustreFileSystemListVO file_system){
		Map<String, Object> result = new HashMap<>();
		result.put("status", lustreService.setFileSystem(file_system));
		return result;
	}

	
	/**
	 * set system
	 * @param file_system
	 * @return
	 */
	@RequestMapping(value="/api/v1/lustre/add_filesystem_and_add_MGTDisk",method= {RequestMethod.GET,RequestMethod.POST})
	public @ResponseBody Map<String, Object> add_filesystem_and_add_MGTDisk(
			LustreFileSystemListVO file_system,
			DiskInforVO disk_info
			){
		String radom_key = IdGeneration.generateId(16);
		Map<String, Object> result = new HashMap<>();
		result.put("status", lustreService.add_filesystem_and_add_MGTDisk(file_system, disk_info,radom_key));
		return result;
	}
	
	/**
     * mds 디스크 추가
     * 181227 je.kim 파일시스템 항목 추가
     * @param lustreNodesVO
     * @param diskInforVO
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/mdsAddDisk_fs",method = {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> mdsAddDisk(
    		LustreNodesVO lustreNodesVO,
    		DiskInforVO diskInforVO,
    		LustreFileSystemListVO lustre_fs
    		){
    	Map<String,Object> result = new HashMap<>();
    	String radom_key = IdGeneration.generateId(16);
    	
		// shkim20181215
		result.put("data", radom_key);
		result.put("status", lustreService.mdsAddDisk(lustreNodesVO, diskInforVO, lustre_fs, radom_key));

    	return result;
    }
    
    /**
     * ost 추가
     * @param lustreNodesVO list 을 통하여 json 데이터를 받아옴
     * @param LustreFileSystemListVO 참고할 파일시스템정보
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/ossAddDisk_fs",method = {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> ossAddDisk(
    		@RequestBody LustreNodesVO lustreNodesVO
    		//,@RequestBody LustreFileSystemListVO lustre_fs
    		){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	// 디비에서 가져오기
    	//lustre_fs = lustreService.viewFileSystem(lustre_fs);
    	
    	if(lustreNodesVO.getList().size() > 0) {
    		List<LustreNodesVO> lustre_list = lustreNodesVO.getList();
    		
    		LustreFileSystemListVO lustre_fs = new LustreFileSystemListVO();
			lustre_fs.setNum(lustreNodesVO.getFile_system_num());
			lustre_fs = lustreService.viewFileSystem(lustre_fs);
    		
    		for (LustreNodesVO lustre : lustre_list) {
    			if(lustre.getDisk_list().size() > 0) {
    				Map<String, Object> tmp = new HashMap<>();
        			String radom_key = IdGeneration.generateId(16);
    				try {
        				tmp.put("data", radom_key);
        				tmp.put("status", lustreService.ossAddDisk(lustre,lustre_fs,radom_key));
        			} catch (Exception e) {
        				tmp.put("status", false);
        				tmp.put("data", e.getMessage());
        			}
        			result.add(tmp);
    			}
    			
			}
    	}
    	
    	
    	return result;
    }
    
    /**
     * update client folder
     * @param clientNodes
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/clientMountFolderSetting",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Boolean clientMountFolderSetting(@RequestBody LustreNodesVO clientNodes){
    	LustreFileSystemListVO fs_data = new LustreFileSystemListVO();
    	fs_data.setNum(clientNodes.getFile_system_num());
    	fs_data = lustreService.viewFileSystem(fs_data);
    	return lustreService.clientMountFolderSetting(clientNodes,fs_data);
    }
    
    
    /**
     * CLient 셋팅
     * @param client_network
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/clientSetting_fs",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> clientSetting(@RequestBody LustreNodesVO clientNodes){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	LustreFileSystemListVO fs_data = new LustreFileSystemListVO();
    	fs_data.setNum(clientNodes.getFile_system_num());
    	fs_data = lustreService.viewFileSystem(fs_data);
    	
    	List<LustreNodesVO> node_list = clientNodes.getList();
    	for (LustreNodesVO lustreNode : node_list) {
    		Map<String, Object> tmp = new HashMap<>();
    		if(lustreNode.getHost_name() == null || lustreNode.getHost_name().equals("")) {
    			tmp.put("data", "node host setting");
				tmp.put("status", false);
    		}else {
    			String radom_key = IdGeneration.generateId(16);
    			try {
    				tmp.put("data", radom_key);
    				tmp.put("status", lustreService.clientSetting(lustreNode,fs_data,radom_key));
				} catch (Exception e) {
					tmp.put("data", e.getMessage());
					tmp.put("status", false);
				}
    		}
    		result.add(tmp);
		}
    	    	
    	return result;
    	
    }
    
    /**
     * file system 추가전에 디스크정보 읽어오
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/getAmbariForMDSDisks",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> getAmbariForMDSDisks(){
    	Map<String,Object> result = new HashMap<>();
    	result.put("result", lustreService.getAmbariForMDSDisks());
    	return result;
    }
    
    
    /**
     * MGT 디스크 추가
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/add_MGS_Disk",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> add_MGS_Disk(){
    	Map<String,Object> result = new HashMap<>();
    	String radom_key = IdGeneration.generateId(16);
    	
    	result.put("data", radom_key);
    	result.put("status", true);

    	return result;
    }
    
    /**
     * 설치된 MGT 디스크 정보 가져오기
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/get_MGT_Disk",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String,Object> get_MGT_Disk(){
    	Map<String,Object> result = new HashMap<>();
    	result.put("data", lustreService.get_MGT_Disk());
    	return result;
    }
    
    
    /**
     * 클라이언트 마운트
     * @param luste_file_system
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/mountClientFolder",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String,Object> mountClientFolder(LustreFileSystemListVO luste_file_system){
    	Map<String,Object> result = new HashMap<>();
    	// row_key 키값 생성
    	String row_key = IdGeneration.generateId(16);
    	// 러스터 파일시스템 정보가져오기
    	luste_file_system = lustreService.viewFileSystem(luste_file_system);
    	
    	result.put("data", row_key);
    	result.put("status", lustreService.mountClientFolder(luste_file_system,row_key));
    	return result;
    }
    
    /**
     * 클라이언트 언마운트
     * @param luste_file_system
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/umountClientFolder",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String,Object> umountClientFolder(LustreFileSystemListVO luste_file_system){
    	Map<String,Object> result = new HashMap<>();
    	// row_key 키값 생성
    	String row_key = IdGeneration.generateId(16);
    	// 러스터 파일시스템 정보가져오기
    	luste_file_system = lustreService.viewFileSystem(luste_file_system);

    	result.put("data", row_key);
    	result.put("status", lustreService.umountClientFolder(luste_file_system,row_key));
    	return result;
    }
    
    
    /**
     * 디스크 백업
     * @param disk_list
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/BackupDisks",method = {RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> backupDisk(@RequestBody List<DiskInforVO> disk_list){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	for (DiskInforVO disk : disk_list) {
    		String radom_key = IdGeneration.generateId(16);
    		Map<String, Object> tmp = new HashMap<>();
    		LustreFileSystemListVO lustre_fs = new LustreFileSystemListVO();
    		lustre_fs.setNum(disk.getFile_system_num());
    		lustre_fs = lustreService.viewFileSystem(lustre_fs);
    		try {
    			tmp.put("status", lustreService.no_async_backupDisk(disk,lustre_fs,radom_key));
    			tmp.put("data", radom_key);
        		result.add(tmp);
			} catch (Exception e) {
				tmp.put("status", false);
				tmp.put("data", e.getMessage());
				result.add(tmp);
			}
    		
    		
		}
    	
    	return result;
    }
    
    /**
     * 디스크 복구
     * @param disk_list
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/RestoreDisks",method = {RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> restoreDisks(@RequestBody List<DiskInforVO> disk_list){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	for (DiskInforVO disk : disk_list) {
    		String radom_key = IdGeneration.generateId(16);
    		Map<String, Object> tmp = new HashMap<>();
    		try {
    			LustreFileSystemListVO file_system = new LustreFileSystemListVO();
    			file_system.setNum(disk.getFile_system_num());
    			tmp.put("status", lustreService.no_async_restoreDisk(disk,file_system,radom_key));
    			tmp.put("data", radom_key);
        		result.add(tmp);
    		} catch (Exception e) {
    			tmp.put("status", false);
				tmp.put("data", e.getMessage());
				result.add(tmp);
    		}
    	}
    	
		return result;
    }
    
    /**
     * Remove OST
     * @param host_name
     * @param disk_name
     * @param index
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/removeOST",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> removeOST(
    		@RequestParam(value="host_name",required=true) String host_name
    		,@RequestParam(value="disk_name",required=true) String disk_name
    		,@RequestParam(value="index",required=true) int index
    		,@RequestParam(value="file_system_num",required=true) Long file_system_num
    		){
    	Map<String,Object> result = new HashMap<>();
    	
    	LustreNodesVO lustreNodesVO = new LustreNodesVO();
    	DiskInforVO diskInforVO = new DiskInforVO();
    	
    	
    	lustreNodesVO.setHost_name(host_name);
    	lustreNodesVO.setNode_type("OSS");
    	lustreNodesVO.setFile_system_num(file_system_num);
    	diskInforVO.setDisk_name(disk_name);
    	diskInforVO.setDisk_type("OST");
    	diskInforVO.setIndex(index);
    	
    	LustreFileSystemListVO file_system = new LustreFileSystemListVO();
    	file_system.setNum(file_system_num);
    	file_system = lustreService.viewFileSystem(file_system);
    	
    	try {
    		String radom_key = IdGeneration.generateId(16);
    		lustreService.no_async_removeOST(lustreNodesVO,diskInforVO,file_system,radom_key);
			result.put("data", radom_key);
			result.put("status", true);
		} catch (TaskRejectedException e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	
    	return result;
    }
    
    
    /**
     * 디비에 있는 디스크 정보가져오기
     * @param diskInforVO
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/getDiskList",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody List<DiskInforVO> getDiskList(DiskInforVO diskInforVO){
    	return lustreService.getDisks(diskInforVO);
    }
    
    /**
     * remove lustre file system
     * @param file_system
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/removeLustreFilesystem",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> removeLustreFilesystem(LustreFileSystemListVO file_system){
    	return lustreService.removeLustreFilesystem(file_system);
    }
    
    /**
     * lnet network stop
     * @param host_name
     * @param node_type
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/networkStop_fs",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> networkStop_fs(
    		@RequestParam(value="host_name",required=true) String host_name,
    		@RequestParam(value="node_type",required=true) String node_type,
    		@RequestParam(value="file_system_num",required=true) String file_system_num
    		){
    	Map<String, Object> result = new HashMap<>();
    	LustreNodesVO search_node = new LustreNodesVO();
    	search_node.setHost_name(host_name);
    	search_node.setNode_type(node_type);
    	search_node.setFile_system_num(Long.parseLong(file_system_num));
    	
    	try {
    		String radom_key = IdGeneration.generateId(16);
			result.put("data", radom_key);
			result.put("status", lustreService.no_async_networkStop(search_node,radom_key));
		} catch (Exception e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	return result;
    }
    
    /**
     * lnet network start
     * @param host_name
     * @param node_type
     * @param conf_file_data
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/networkStart_fs",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> networkStart(
    		@RequestParam(value="host_name",required=true) String host_name,
    		@RequestParam(value="node_type",required=true) String node_type,
    		@RequestParam(value="conf_file_data",required=true) String conf_file_data,
    		@RequestParam(value="file_system_num",required=true) String file_system_num
    		){
    	Map<String, Object> result = new HashMap<>();
    	
    	LustreNodesVO search_node = new LustreNodesVO();
    	search_node.setHost_name(host_name);
    	search_node.setNode_type(node_type);
    	search_node.setFile_system_num(Long.parseLong(file_system_num));
    	try {
    		String radom_key = IdGeneration.generateId(16);
			result.put("data", radom_key);
			result.put("status", lustreService.no_async_networkStart(search_node,conf_file_data,radom_key));
		} catch (Exception e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	return result;
    }
    
    /**
     * lnet network all stop
     * @param list
     * @return
     */
    @RequestMapping(
    		value="/api/v1/lustre/networkAllStop_fs"
    		,method= {RequestMethod.GET,RequestMethod.POST}
    		,consumes="application/json"
    		,produces="application/json"
    		)
    public @ResponseBody List<Map<String, Object>> networkAllStop(@RequestBody List<Map<String, Object>> list){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	for (Map<String, Object> node : list) {
    		LustreNodesVO search_node = new LustreNodesVO();
    		Map<String, Object> tmp_map = new HashMap<>();
    		
    		String host_name = (String) node.get("host_name");
    		String node_type = (String) node.get("node_type");
    		
        	search_node.setHost_name(host_name);
        	search_node.setNode_type(node_type);
        	
        	try {
        		String radom_key = IdGeneration.generateId(16);
    			tmp_map.put("data", radom_key);
    			tmp_map.put("status", lustreService.no_async_networkStop(search_node,radom_key));
    		} catch (TaskRejectedException e) {
    			tmp_map.put("status", false);
    			tmp_map.put("data", e.getMessage());
    		}
        	result.add(tmp_map);
		}
    	
    	return result;
    }
    
    /**
     * lnet network all start
     * @param list
     * @return
     */
    @RequestMapping(
    		value="/api/v1/lustre/networkAllStart_fs"
    		,method= {RequestMethod.GET,RequestMethod.POST}
    		,consumes="application/json"
    		,produces="application/json"
    		)
    public @ResponseBody List<Map<String, Object>> networkAllStart(@RequestBody List<Map<String, Object>> list){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	for (Map<String, Object> node : list) {
    		LustreNodesVO search_node = new LustreNodesVO();
    		Map<String, Object> tmp_map = new HashMap<>();
    		
    		String host_name = (String) node.get("host_name");
    		String node_type = (String) node.get("node_type");
    		String file_system_num = (String) node.get("file_system_num");
    		String conf_file_data = (String) node.get("data");
    		
        	search_node.setHost_name(host_name);
        	search_node.setNode_type(node_type);
        	search_node.setFile_system_num(Long.parseLong(file_system_num));
        	
        	try {
        		String radom_key = IdGeneration.generateId(16);
    			tmp_map.put("data", radom_key);
    			tmp_map.put("status", lustreService.no_async_networkStart(search_node,conf_file_data,radom_key));
    		} catch (TaskRejectedException e) {
    			tmp_map.put("status", false);
    			tmp_map.put("data", e.getMessage());
    		}
        	result.add(tmp_map);
		}
    	
    	return result;
    }
}

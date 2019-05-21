package com.xiilab.lustre.api;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Resource;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.task.TaskRejectedException;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import com.xiilab.lustre.async.AsyncTaskService;
import com.xiilab.lustre.configs.AsyncConfig;
import com.xiilab.lustre.model.DiskInforVO;
import com.xiilab.lustre.model.LustreFileSystemListVO;
import com.xiilab.lustre.model.LustreNodesVO;
import com.xiilab.lustre.utilities.IdGeneration;

/**
 * 비동기 전용 컨트롤러
 * @author xiilab
 *
 */
@Controller("AsyncController")
public class AsyncController {
	
	@Autowired
	private LustreService lustreService;
	
	
	/** 비동기 처리를 위한 서비스 */
    @Resource(name = "asyncTaskService")
    private AsyncTaskService asyncTaskLustreManager;
 
    /** 비동기 처리 설정 */
    @Resource(name = "asyncConfig")
    private AsyncConfig asyncConfig;

    @RequestMapping("/testAsync")
    public @ResponseBody Map<String,Object> doTask(HttpServletRequest request, HttpServletResponse response) throws Exception {
    	Map<String,Object> result = new HashMap<>();
    	
    	//////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////
		try {
			// 등록 가능 여부 체크
			if (asyncConfig.isSampleTaskExecute()) {
				// task 사용
				asyncTaskLustreManager.executorSample("ㄱ");
				result.put("status", "1qwes23");
			} else {
				System.out.println("==============>>>>>>>>>>>> THREAD 개수 초과");
				result.put("status", false);
			}
		} catch (TaskRejectedException e) {
			// TaskRejectedException : 개수 초과시 발생
			System.out.println("==============>>>>>>>>>>>> THREAD ERROR");
			System.out.println("TaskRejectedException : 등록 개수 초과");
			System.out.println("==============>>>>>>>>>>>> THREAD END");
			result.put("status", false);
		}
		return result;
    }
    
    
    /**
     * mds 디스크 추가
     * @param lustreNodesVO
     * @param diskInforVO
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/mdsAddDisk",method = {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> mdsAddDisk(LustreNodesVO lustreNodesVO,DiskInforVO diskInforVO,LustreFileSystemListVO lustre_fs){
    	Map<String,Object> result = new HashMap<>();
    	String radom_key = IdGeneration.generateId(16);
    	
    	try {
			// 등록 가능 여부 체크
			if (asyncConfig.isSampleTaskExecute()) {
				// task 사용
				// shkim20181212 test후 주석 
				//asyncTaskLustreManager.mdsAddDisk(lustreNodesVO, diskInforVO, lustre_fs, radom_key);
				result.put("data", radom_key);
				result.put("status", true);
			} else {
				result.put("status", false);
				result.put("data", "개수 초과");
			}
		} catch (TaskRejectedException e) {
			// TaskRejectedException : 개수 초과시 발생
//			System.out.println("==============>>>>>>>>>>>> THREAD ERROR");
//			System.out.println("TaskRejectedException : 등록 개수 초과");
//			System.out.println("==============>>>>>>>>>>>> THREAD END");
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	return result;
    }
    
    /**
     * ost 추가
     * @param lustreNodesVO
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/ossAddDisk",method = {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> ossAddDisk(@RequestBody LustreNodesVO lustreNodesVO){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	if(lustreNodesVO.getList().size() > 0) {
    		List<LustreNodesVO> lustre_list = lustreNodesVO.getList();
    		for (LustreNodesVO lustre : lustre_list) {
    			
    			if(lustre.getDisk_list().size() > 0) {
    				Map<String, Object> tmp = new HashMap<>();
        			String radom_key = IdGeneration.generateId(16);
    				try {
        				//asyncTaskLustreManager.ossAddDisk(lustre,radom_key);
        				tmp.put("data", radom_key);
        				tmp.put("status", true);
        			} catch (TaskRejectedException e) {
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
     * ost deactivate
     * @param host_name 
     * @param disk_name
     * @param index
     * @param disk_type
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/ostDeactivate",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> deactivate(
    		@RequestParam(value="host_name",required=true) String host_name
    		,@RequestParam(value="disk_name",required=true) String disk_name
    		,@RequestParam(value="index",required=true) int index
    		,@RequestParam(value="file_system_num",required=true) Long file_system_num
    		){
    	Map<String,Object> result = new HashMap<>();
    	
    	LustreNodesVO lustreNodesVO = new LustreNodesVO();
    	DiskInforVO diskInforVO = new DiskInforVO();
    	
    	LustreFileSystemListVO file_system = new LustreFileSystemListVO();
    	file_system.setNum(file_system_num);
    	file_system = lustreService.viewFileSystem(file_system);
    	
    	lustreNodesVO.setHost_name(host_name);
    	lustreNodesVO.setFile_system_num(file_system_num);
    	lustreNodesVO.setNode_type("OSS");
    	diskInforVO.setDisk_name(disk_name);
    	diskInforVO.setDisk_type("OST");
    	diskInforVO.setIndex(index);
    	
    	try {
    		String radom_key = IdGeneration.generateId(16);
			asyncTaskLustreManager.deactivate(lustreNodesVO,diskInforVO,file_system,radom_key);
			result.put("data", radom_key);
			result.put("status", true);
		} catch (TaskRejectedException e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	
    	return result;
    }
    
    
    /**
     * ost activate
     * @param host_name
     * @param disk_name
     * @param index
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/ostActivate",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> activate(
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
			asyncTaskLustreManager.activate(lustreNodesVO,diskInforVO,file_system,radom_key);
			result.put("data", radom_key);
			result.put("status", true);
		} catch (TaskRejectedException e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	
    	return result;
    	
    }
    
    
    /**
     * CLient 셋팅
     * @param client_network
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/clientSetting_bak",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> clientSettingBak(
    		@RequestParam(value="client_network",required=true) String client_network,
    		@RequestParam(value="client_mount_point",required=true) String client_mount_point
    		){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	LustreNodesVO search_node = new LustreNodesVO();
    	search_node.setNode_type("CLIENT");
    	
    	List<LustreNodesVO> lustreNodesVO = lustreService.getLustreNodes(search_node);
    	for (LustreNodesVO lustre_node : lustreNodesVO) {
    		Map<String, Object> tmp = new HashMap<>();
			String radom_key = IdGeneration.generateId(16);
			lustre_node.setNetwork_device(client_network);
			lustre_node.setLustre_client_folder(client_mount_point);
			try {
				asyncTaskLustreManager.clientSetting(lustre_node,radom_key);
				tmp.put("data", radom_key);
				tmp.put("status", true);
			} catch (TaskRejectedException e) {
				tmp.put("status", false);
				tmp.put("data", e.getMessage());
			}
			result.add(tmp);
		}
    	
		
    	
    	return result;
    	
    }
    
    /**
     * CLient 셋팅
     * @param client_network
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/clientSetting",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody List<Map<String, Object>> clientSetting(@RequestBody LustreNodesVO clientNodes){
    	List<Map<String, Object>> result = new ArrayList<>();
    	
    	
    	List<LustreNodesVO> node_list = clientNodes.getList();
    	for (LustreNodesVO lustreNode : node_list) {
    		Map<String, Object> tmp = new HashMap<>();
    		if(lustreNode.getHost_name() == null || lustreNode.getHost_name().equals("")) {
    			tmp.put("data", "node host setting");
				tmp.put("status", false);
    		}else {
    			String radom_key = IdGeneration.generateId(16);
    			try {
    				//asyncTaskLustreManager.clientSetting(lustreNode,radom_key);
    				tmp.put("data", radom_key);
    				tmp.put("status", true);
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
     * lnet network start
     * @param host_name
     * @param node_type
     * @param conf_file_data
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/networkStart",method= {RequestMethod.GET,RequestMethod.POST})
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
			asyncTaskLustreManager.networkStart(search_node,conf_file_data,radom_key);
			result.put("data", radom_key);
			result.put("status", true);
		} catch (TaskRejectedException e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	return result;
    }
    
    
    
    /**
     * lnet network stop
     * @param host_name
     * @param node_type
     * @return
     */
    @RequestMapping(value="/api/v1/lustre/networkStop",method= {RequestMethod.GET,RequestMethod.POST})
    public @ResponseBody Map<String, Object> networkStop(
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
			asyncTaskLustreManager.networkStop(search_node,radom_key);
			result.put("data", radom_key);
			result.put("status", true);
		} catch (TaskRejectedException e) {
			result.put("status", false);
			result.put("data", e.getMessage());
		}
    	
    	return result;
    }
    
    
    
    /**
     * lnet network all start
     * @param list
     * @return
     */
    @RequestMapping(
    		value="/api/v1/lustre/networkAllStart"
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
    			asyncTaskLustreManager.networkStart(search_node,conf_file_data,radom_key);
    			tmp_map.put("data", radom_key);
    			tmp_map.put("status", true);
    		} catch (TaskRejectedException e) {
    			tmp_map.put("status", false);
    			tmp_map.put("data", e.getMessage());
    		}
        	result.add(tmp_map);
		}
    	
    	return result;
    }
    
    
    
    /**
     * lnet network all stop
     * @param list
     * @return
     */
    @RequestMapping(
    		value="/api/v1/lustre/networkAllStop"
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
    			asyncTaskLustreManager.networkStop(search_node,radom_key);
    			tmp_map.put("data", radom_key);
    			tmp_map.put("status", true);
    		} catch (TaskRejectedException e) {
    			tmp_map.put("status", false);
    			tmp_map.put("data", e.getMessage());
    		}
        	result.add(tmp_map);
		}
    	
    	return result;
    }
    
   
    
}

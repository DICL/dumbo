package com.xiilab.app_monitor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.xiilab.ambari.AmbariStatusDAO;
import com.xiilab.mapper1.ApplicationMonitorMapper;
import com.xiilab.mapper2.AmbariMapper;
import com.xiilab.models.YarnAppMonitorPerNodeVO;
import com.xiilab.models.YarnAppMonitorVO;

@Repository
public class ApplicationMonitoringDAO {
	
	@Autowired
	private AmbariStatusDAO ambariStatusDAO;
	@Autowired
	private ApplicationMonitorMapper applicationMonitorRepository;
	@Autowired
	private AmbariMapper ambariMapper;

	public List<Map<String, Object>> getYarnJobMonitor() {
		List<Map<String, Object>> result = new ArrayList<>();
		
		// yarn api 에서 FINISHED 을 제외한 리스트 가져오기
		// application id 만 추출
		
		List<String> application_id_list = new ArrayList<>();
		
		String RunningYarnJobs = ambariStatusDAO.yarnAppList("NEW,NEW_SAVING,SUBMITTED,ACCEPTED,RUNNING");
		ObjectMapper mapper = new ObjectMapper();
		try {
			JsonNode actualObj = mapper.readTree(RunningYarnJobs);
			
			if(actualObj.get("apps") != null && !"null".equals(actualObj.get("apps").asText()) ) {
				Iterator application_list =  actualObj.get("apps").get("app").elements();
				while(application_list.hasNext()){
					JsonNode application_obj = (JsonNode) application_list.next();
					application_id_list.add(application_obj.get("id").asText());
				}
				//result = tmp_list;
			}
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		
		for (String application_id : application_id_list) {
			// application_id 이용하여 container_id 추출
			List<String> container_id_list = applicationMonitorRepository.getContainerIdForApplicationMonitor(application_id);
			for (String container_id : container_id_list) {
				String rowkey = applicationMonitorRepository.getRowKeyForApplicationMonitor(container_id);
				YarnAppMonitorVO seachData = new YarnAppMonitorVO();
				seachData.setRowkey(rowkey);
				List<YarnAppMonitorVO> yarnAppMonitorList = applicationMonitorRepository.getYarnAppMonitorListBak(seachData);
				
				Map<String, Object> tmpMap = new HashMap<>();
				if(yarnAppMonitorList.size() > 0) {
					tmpMap.put("create_date", yarnAppMonitorList.get(0).getCreate_date());
					tmpMap.put("pid", yarnAppMonitorList.get(0).getPid());
					tmpMap.put("application_id", yarnAppMonitorList.get(0).getApplication_id());
					tmpMap.put("container_id", yarnAppMonitorList.get(0).getContainer_id());
					tmpMap.put("node", yarnAppMonitorList.get(0).getNode());
				}
				for (YarnAppMonitorVO yarnAppMonitorVO : yarnAppMonitorList) {
					tmpMap.put(yarnAppMonitorVO.getMetric(), yarnAppMonitorVO.getVal());
				}
				result.add(tmpMap);
			}
		}
		
		return result;
	}

	public List<YarnAppMonitorVO> getYarnJobHistoryBak(YarnAppMonitorVO yarnAppMonitorVO) {
		return applicationMonitorRepository.getYarnAppMonitorListBak(yarnAppMonitorVO);
	}
	public List<Map<String, Object>> getYarnJobHistory(YarnAppMonitorVO yarnAppMonitorVO) {
		
		String hyper_table_name = applicationMonitorRepository.findHyperTable(yarnAppMonitorVO);
		
		if(hyper_table_name != null ) {
			Map<String,Object> find_data = new HashMap<>();
			find_data.put("table_name", hyper_table_name);
			
			return applicationMonitorRepository.getYarnAppMonitorList(find_data);
		}else {
			List<Map<String, Object>> tmpdata = new ArrayList<>();
			return tmpdata; 
		}
		
	}
	
	public List<Map<String, Object>> getYarnJobHistoryPerNode(YarnAppMonitorPerNodeVO yarnAppMonitorPerNodeVO) {
		
		List<Map<String, Object>> hyper_table_name = applicationMonitorRepository.findHyperTablePerNode(yarnAppMonitorPerNodeVO);
		
		if(hyper_table_name.size() > 0 ) {
			List<Map<String, Object>> find_data = new ArrayList<Map<String, Object>>();
			for(int i = 0; i < hyper_table_name.size(); i++){
				Map<String,Object> tmp_data = new HashMap<>();
				tmp_data.put("table_name", ((Map<String,Object>)hyper_table_name.get(i)).get("table_name"));
				tmp_data.put("node", yarnAppMonitorPerNodeVO.getNode());
				find_data.addAll(applicationMonitorRepository.getYarnAppMonitorListPerNode(tmp_data));
			}	
			
			return find_data;
		}else {
			List<Map<String, Object>> tmpdata = new ArrayList<>();
			return tmpdata; 
		}
		
	}

	/**
	 * Metric Registry View 에 저장되어 있는 Metric list 가져오기 
	 * @return
	 */
	public List<Map<String, Object>> getMetricList() {
		return ambariMapper.getMetricList();
	}

}

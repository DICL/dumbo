package com.xiilab.ldap.config;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class LdapService {

	@Autowired
	private LdapDAO ldapDAO;
	
	/**
	 * 리스트를 조회하는 서비스매서드
	 * @return
	 * @throws IOException
	 */
	public Map<String,Object> getUserListTree() throws IOException{
		Map<String, Object> result = new HashMap<String, Object>();
		result.put("status", true);
		result.put("result", ldapDAO.getUserListTree());
		return result;
	}
	
	public List<Map<String, Object>> getUserInfo(String userId) throws IOException  {
		return ldapDAO.getUserInfo(userId);
	}

	
	/**
	 * 유저정보를 생성하는 서비스매서드
	 * @param sendData
	 * @return
	 * @throws IOException
	 */
	public Map<String, Object> addUser(List<Map<String, Object>> sendData) throws IOException {
		Map<String, Object> result = new HashMap<String, Object>();
		//잘못된 결로 입력시
		if(sendData == null) {
			Map<String, Object> falseData = new HashMap<String, Object>();
			falseData.put("status", false);
			falseData.put("message", "Invalid Value.");
			return falseData;
		}
		//값을 입력안했을때
		for (Map<String, Object> map : sendData) {
			if(map.get("value").equals("")) {
				Map<String, Object> falseData = new HashMap<String, Object>();
				falseData.put("status", false);
				falseData.put("message", "Enter value.");
				return falseData;
			}
		}
		
		result.put("result", ldapDAO.addUser(sendData));
		result.put("status", true);
		return result;
	}

	// 유저삭제 (JSON)
	public Map<String, Object> deleteUser(Map<String, Object> sendData) throws IOException{
		Map<String, Object> result = new HashMap<String, Object>();
		if(sendData == null) {
			Map<String, Object> falseData = new HashMap<String, Object>();
			falseData.put("status", false);
			falseData.put("message", "Invalid Value.");
			return falseData;
		}
		result.put("result", ldapDAO.deleteUser(sendData));
		result.put("status", true);
		return result;
	}


	//서버정보
	public Map<String, String> serverInfo() {
		return ldapDAO.serverInfo();
	}

	/**
	 * uidNumber, gidNumber 값을 출력한다
	 * @return
	 */
	public Map<String, Object> getMaxUidNumber() {
		Map<String, Object> result = new HashMap<String, Object>();
		try {
			result.put("result", ldapDAO.getMaxUidNumber("hadoop"));
			result.put("status", true);
		} catch (IOException e) {
			result.put("status", false);
		}
		
		return result;
	}

	/**
	 * 하둡 홈디렉토리 생성
	 * @return Map<String, Object>
	 */
	public Map<String, Object> craetedHadoopHomeFolder(String userId) {		
		try {
			return ldapDAO.craetedHadoopHomeFolder(userId);
		} catch (IOException e) {
			e.printStackTrace();
			Map<String, Object> result = new HashMap<String, Object>();
			result.put("status", false);
			result.put("result", "system error");
			return result;
		}
	}
	
}

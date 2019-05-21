package com.xiilab.ldap.config;

import java.io.BufferedReader;

import org.springframework.beans.factory.annotation.Autowired;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.springframework.stereotype.Repository;

import com.xiilab.ldap.utilities.AmbariProperty;


/**
 * Open LDAP 명령어들을 처리하는 DAO
 * @author xiilab
 *
 */
@Repository
public class LdapDAO {
	protected Log log = LogFactory.getLog(LdapDAO.class);
	
	
	@Autowired
	private HttpServletRequest request;
	
	@Autowired
	private AmbariProperty ambariProperty;
	
	
/*	
 	[
     { "id" : "ajson1", "parent" : "#", "text" : "Simple root node" },
     { "id" : "ajson2", "parent" : "#", "text" : "Root node 2" },
     { "id" : "ajson3", "parent" : "ajson2", "text" : "Child 1" },
     { "id" : "ajson4", "parent" : "ajson2", "text" : "Child 2" },
    ]
*/
	
	/**
	 * 리스트를 조회한다
	 * @param context_path 서블릿 루트폴더 홈위치
	 * @return
	 * @throws IOException
	 */
	public List<Map<String,Object>> getUserListTree() throws IOException{
		
		String context_path = request.getServletContext().getRealPath("/");
		List<Map<String,Object>> result = new ArrayList<Map<String,Object>>();
		//초기변수
//		String sshPassWord = this.sshPassWord;
//		String domain =  this.domain;
//		String dc =  this.dc;
		
		String sshPassWord = ambariProperty.getServerConfigs().get("sshPassWord");
		String domain = ambariProperty.getServerConfigs().get("domain");
		String dc = ambariProperty.getServerConfigs().get("dc");
		
		// 쉘입력값
		String[] argument = {
				sshPassWord,
				domain,
				dc,
		};
		
		//스크립트 파일 위치 추적
		String shell_path = context_path + "/WEB-INF/classes/shell_scripts/list.sh";
		
		//스크립트 파일 실행
		Process proc = Runtime.getRuntime().exec("sh " + shell_path+" "+String.join(" ", argument)); 
				
		StringBuffer resultString = new StringBuffer(); 
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
				 
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			resultString.append(inputLine+"\n");
		}
		// 출력된 결과을 파싱
		// # 문자로 배열처리
		String[] result_array = resultString.toString().split("#");
		for (int i = 0; i < result_array.length; i++) {
			// 개행문자로 구분(2줄이상일때)
			if(result_array[i].split("\n").length > 3) {
				Map<String,Object> hashmap = new HashMap<String, Object>();
				String[] result_array2 = result_array[i].split("\n");
				for (int j = 0; j < result_array2.length; j++) {
					//도메인
					if(result_array2[0].split(",").length < 2) {
						hashmap.put("id", result_array2[1].replaceAll("dn: ", ""));
						hashmap.put("parent","#");
						hashmap.put("text", result_array2[0].split(",")[0]);
						hashmap.put("icon", "glyphicon glyphicon-home");
					}
					//그룹
					else if(result_array2[0].split(",").length == 2) {
						hashmap.put("id", result_array2[1].split(",")[0].replaceAll("dn: ", ""));
						hashmap.put("parent",result_array2[1].split(",")[1]+","+result_array2[1].split(",")[2]);
						hashmap.put("text", result_array2[1].split(",")[0].replaceAll("dn: ", "").split("=")[1]);
					}
					//사용자
					else if(result_array2[0].split(",").length > 2) {
						hashmap.put("id", result_array2[1].split(",")[0].replaceAll("dn: ", ""));
						hashmap.put("parent",result_array2[1].split(",")[1]);
						hashmap.put("text", result_array2[1].split(",")[0].replaceAll("dn: ", "").split("=")[1]);
						hashmap.put("icon", "glyphicon glyphicon-user");
					}
					//uid & gid & dn 삽입
					if( result_array2[j].split(":").length > 1) {
						if(result_array2[j].split(":")[0].contains("uidNumber")) {
							hashmap.put("uidNumber",result_array2[j].split(":")[1].trim());
						}
						if(result_array2[j].split(":")[0].contains("gidNumber")) {
							hashmap.put("gidNumber",result_array2[j].split(":")[1].trim());
						}
						if(result_array2[j].split(":")[0].contains("dn")) {
							hashmap.put("dn",result_array2[j].split(":")[1].trim());
						}
					}
				}
				result.add(hashmap);
			}
		}
				
		return result;
	}
	
	
	/**
	 * 상세정보를 조회한다
	 * @param context_path 서블릿 루트폴더 홈위치
	 * @param userId	유저아이디
	 * @return
	 * @throws IOException
	 */
	public List<Map<String,Object>> getUserInfo(String userId) throws IOException {
		
		String context_path = request.getServletContext().getRealPath("/");
		
		List<Map<String,Object>> result = new ArrayList<Map<String,Object>>();
		if(userId == null || "".equals(userId)) return null;
		
		//초기변수
//		String sshPassWord = this.sshPassWord;
//		String domain = this.domain;
//		String dc = this.dc;
		
		String sshPassWord = ambariProperty.getServerConfigs().get("sshPassWord");
		String domain = ambariProperty.getServerConfigs().get("domain");
		String dc = ambariProperty.getServerConfigs().get("dc");
		
		String[] argument = {
				sshPassWord,
				domain,
				dc,
				userId,
		};
		
		
		
		//스크립트 파일 위치 추적
		String shell_path = context_path + "/WEB-INF/classes/shell_scripts/view.sh";

		
		//스크립트 파일 실행
		Process proc = Runtime.getRuntime().exec("sh " + shell_path+" "+String.join(" ", argument));
		StringBuffer resultString = new StringBuffer(); 
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
				 
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			resultString.append(inputLine+"\n");
		}
		
		// # 문자로 배열처리
		String[] result_array = resultString.toString().split("#");
		for (int i = 0; i < result_array.length; i++) {
			// 개행문자로 구분
			if(result_array[i].split("\n").length > 3 && result_array[i].contains(userId)) {
				String[] result_array2 = result_array[i].split("\n");
				for (int j = 0; j < result_array2.length; j++) {
					// :: 문자만 구별
					Map<String,Object> hashmap = new HashMap<String, Object>();
					if(result_array2[j].split("::").length == 2) {
						hashmap.put("name", result_array2[j].split("::")[0]);
						hashmap.put("value", result_array2[j].split("::")[1]);
						//hashmap.put("value", "");
						hashmap.put("type", "password");
						result.add(hashmap);
					}
					// : 문자만 구별
					if(result_array2[j].split(":").length == 2) {
						hashmap.put("name", result_array2[j].split(":")[0]);
						hashmap.put("value", result_array2[j].split(":")[1].trim());
						hashmap.put("type", "text");
						result.add(hashmap);
					}
					
				}
			}
		}
		return result;
	}

	/**
	 * 사용자를 추가한다
	 * @param context_path 서블릿 루트폴더 홈위치
	 * @param sendData	웹에서 전송받은 사용자 정보
	 * @return
	 * @throws IOException
	 */
	public String addUser(List<Map<String, Object>> sendData) throws IOException {
		
		//초기변수
//		String sshPassWord = this.sshPassWord;
//		String domain = this.domain;
//		String ManagerPassword = this.ManagerPassword;
//		String dc = this.dc;
		
		String context_path = request.getServletContext().getRealPath("/");
		
		Map<String, String> inputData = new HashMap<String, String>();
		
		String sshPassWord = ambariProperty.getServerConfigs().get("sshPassWord");
		String domain = ambariProperty.getServerConfigs().get("domain");
		String ManagerPassword = ambariProperty.getServerConfigs().get("ManagerPassword");
		String dc = ambariProperty.getServerConfigs().get("dc");
		String ManagerName = ambariProperty.getServerConfigs().get("ManagerName");
		
		
		
		/*
		 * sendData 형식
		 * [
		 * 		{
		 * 			name : key,
		 * 			value : test
		 * 		},
		 * 		{
		 * 			name : password,
		 * 			value : test
		 * 		},
		 * ]
		 */
		for (Map<String, Object> map : sendData) {
			String key = (String) map.get("name");
			if(key.equals("uid")) {
				inputData.put("uid", (String) map.get("value"));
			}else if(key.equals("sn")) {
				inputData.put("sn", (String) map.get("value"));
			}else if(key.equals("cn")) {
				inputData.put("cn", (String) map.get("value"));
			}else if(key.equals("userPassword")) {
				inputData.put("userPassword", getLDAPPassword((String) map.get("value")));
			}else if(key.equals("uidNumber")) {
				inputData.put("uidNumber", (String) map.get("value"));
			}else if(key.equals("gidNumber")) {
				inputData.put("gidNumber", (String) map.get("value"));
			}else if(key.equals("homeDirectory")) {
				inputData.put("homeDirectory", (String) map.get("value"));
			}else if(key.equals("loginShell")) {
				inputData.put("loginShell", (String) map.get("value"));
			}
		}
		
		/*
		 * sshpassword=$1
		 * domain=$2
		 * dc=$3
		 * ManagerPassword=$4
		 * uid=$5
		 * cn=$6
		 * sn=$7
		 * userPassword=$8
		 * loginShell=$9
		 * uidNumber=$10
		 * gidNumber=$11
		 * homeDirectory=$12
		 */
		
		// 쉘입력값
		String[] argument = {
				sshPassWord,
				domain,
				dc,
				ManagerPassword,
				inputData.get("uid"),
				inputData.get("cn"),
				inputData.get("sn"),
				inputData.get("userPassword"),
				inputData.get("loginShell"),
				inputData.get("uidNumber"),
				inputData.get("gidNumber"),
				inputData.get("homeDirectory"),
				ManagerName,
		};
		
		
		List<Map<String,Object>> result = new ArrayList<Map<String,Object>>();
		
		//스크립트 파일 위치 추적
		String shell_path = context_path + "/WEB-INF/classes/shell_scripts/add.sh";
						
		
		//스크립트 파일 실행
		Process proc = Runtime.getRuntime().exec("sh " + shell_path+" "+ String.join(" ", argument));
		StringBuffer resultString = new StringBuffer(); 
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
				 
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			resultString.append(inputLine+"\n");
		}
		
		return resultString.toString();
//		return shell_path +" "+ String.join(" ", argument);
	}

	/**
	 * 사용자를 삭제한다
	 * @param context_path 서블릿 루트폴더 홈위치
	 * @param sendData	웹에서 전송받은 사용자 정보 (dn 값을 받아온다)
	 * @return
	 * @throws IOException
	 */
	public String deleteUser(Map<String, Object> sendData)  throws IOException{
		//초기변수
//		String sshPassWord = this.sshPassWord;
//		String domain = this.domain;
//		String ManagerPassword = this.ManagerPassword;
//		String dc = this.dc;
		
		String context_path = request.getServletContext().getRealPath("/");
		
		String sshPassWord = ambariProperty.getServerConfigs().get("sshPassWord");
		String domain = ambariProperty.getServerConfigs().get("domain");
		String ManagerPassword = ambariProperty.getServerConfigs().get("ManagerPassword");
		String dc = ambariProperty.getServerConfigs().get("dc");
		String ManagerName = ambariProperty.getServerConfigs().get("ManagerName");
		
		String[] argument = {
				sshPassWord,
				domain,
				dc,
				ManagerPassword,
				(String) sendData.get("dn"),
				ManagerName,
		};
		
		
		//스크립트 파일 위치 추적
		String shell_path = context_path + "/WEB-INF/classes/shell_scripts/del.sh";
		
		//스크립트 파일 실행
		Process proc = Runtime.getRuntime().exec("sh " + shell_path+" "+ String.join(" ", argument));
		StringBuffer resultString = new StringBuffer(); 
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
						 
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			resultString.append(inputLine+"\n");
		}
				
		return resultString.toString();
//		return shell_path +" "+ String.join(" ", argument);
	}


	/**
	 * Ambari View 에서 설정한 환경설정 정보를 읽어온다
	 * @return
	 */
	public Map<String, String> serverInfo() {
//		String sshPassWord = this.sshPassWord;
//		String domain = this.domain;
//		String ManagerPassword = this.ManagerPassword;
//		String dc = this.dc;
		
		String domain = ambariProperty.getServerConfigs().get("domain");
		String dc = ambariProperty.getServerConfigs().get("dc");
		String ManagerName = ambariProperty.getServerConfigs().get("ManagerName");
		
		
		Map<String, String> result = new HashMap<String, String>();
		result.put("dc", dc);
		result.put("domain", domain);
		result.put("ManagerName", ManagerName);
		return result;
	}

	
	/**
	 * slappasswd 명령어을 이용하여 패스워드를 출력한다
	 * @param rawPassword 	변경하고자 하는 패스워드
	 * @return				slappasswd 에서 처리된 결과값
	 * @throws IOException
	 */
	public String getLDAPPassword(String rawPassword) throws IOException {
		
		
		String sshPassWord = ambariProperty.getServerConfigs().get("sshPassWord");
		String domain = ambariProperty.getServerConfigs().get("domain");
		String encodingPassword = ""; //출력될 패스워드
		
		String context_path = request.getServletContext().getRealPath("/");
		//스크립트 파일 위치 추적
		String shell_path = context_path + "/WEB-INF/classes/shell_scripts/password.sh";
		
		
		String[] argument = {
				sshPassWord,
				domain,
				rawPassword,
		};
		
		Process proc = Runtime.getRuntime().exec("sh " + shell_path+" "+ String.join(" ", argument));
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
						 
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			if(inputLine.contains("{")) {
				encodingPassword = inputLine;
			}
		}
		return encodingPassword;
	}


	/**
	 * LDAP Server에서 uidNumber, gidNumber 들중에서 최대값을 추출한다.
	 * @return	uidNumber,gidNumber 값들을 해쉬맵으로 출력
	 * {
	 * 		uidNumber : 1027,
	 * 		gidNumber: 925,
	 * }
	 * @throws IOException 
	 */
	public Map<String, Object> getMaxUidNumber(String groupName) throws IOException {
		if(groupName == null) return null;
		Map<String, Object> result = new HashMap<String, Object>();
		
		String sshPassWord = ambariProperty.getServerConfigs().get("sshPassWord");
		String domain = ambariProperty.getServerConfigs().get("domain");
		String dc = ambariProperty.getServerConfigs().get("dc");
		String ManagerPassword = ambariProperty.getServerConfigs().get("ManagerPassword");
		
		String context_path = request.getServletContext().getRealPath("/");
		//스크립트 파일 위치 추적
		String shell_path = context_path + "/WEB-INF/classes/shell_scripts/findMaxUidAndGidNumber.sh";
		
		// argument 순서정리
		String[] argument = {
				sshPassWord,
				domain,
				dc,
				ManagerPassword,
				groupName
		};
		
		Process proc = Runtime.getRuntime().exec("sh " + shell_path+" "+ String.join(" ", argument));
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
		
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			// 첫번째 줄이 uidNumber
			if(i==0) {
				result.put("uidNumber", Integer.parseInt(inputLine));
			// 두번째 줄이 gidNumber
			}else if(i==1) {
				result.put("gidNumber", Integer.parseInt(inputLine));
			}
		}
		result.put("groupName",groupName);
		return result;
	}

	/**
	 * 하둡 홈디렉토리 생성
	 * @return Map<String, Object>
	*/
	public Map<String, Object> craetedHadoopHomeFolder(String userId) throws IOException {
		Map<String, Object> result = new HashMap<String, Object>();


		// Process proc = Runtime.getRuntime().exec(" mkdir /lustre/hadoop/user/"+userId); 
		// 19.01.22 je.kim 
		String command = " hadoop fs -mkdir /user/" + userId;
		Process proc = Runtime.getRuntime().exec(command);
		StringBuffer resultString = new StringBuffer(); 
		String inputLine; 
		BufferedReader inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
				 
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			resultString.append(inputLine+"\n");
		}


		// proc = Runtime.getRuntime().exec(" chown "+userId+". /lustre/hadoop/user/"+userId);
		command = " hadoop fs -chown "+userId+" /user/" + userId;
		proc = Runtime.getRuntime().exec(command);
		inputBuf = new BufferedReader(new InputStreamReader(proc.getInputStream(), "UTF-8"));
		for (int i = 0; ((inputLine = inputBuf.readLine()) != null); i++) {
			resultString.append(inputLine+"\n");
		}

		result.put("status", true);
		result.put("result", resultString.toString());

		return result;
	}
	
}

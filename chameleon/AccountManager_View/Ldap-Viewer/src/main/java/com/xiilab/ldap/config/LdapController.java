package com.xiilab.ldap.config;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

/**
 * 페이지 내에서 모든 Ajax 데이터를 처리한다
 * @author xiilab
 *
 */
@Controller
@RequestMapping(value = "/ldap")
public class LdapController {
	
	@Autowired
	private LdapService ldapService;
	
	
	/**
	 * LDAP Server 에서 등록된 유저정보들을 jsTree에서 보여주기 위하여 Json 형태로 출력이 된다.
	 * @return
	 *  {
	 *      "result": [{
	 *          "parent": "#",
	 *          "icon": "glyphicon glyphicon-home",
	 *          "dn": "dc=srv,dc=world",
	 *          "id": "dc=srv,dc=world",
	 *          "text": " srv.world"
	 *      }, {
	 *          "parent": "ou=People",
	 *          "uidNumber": "1000",
	 *          "icon": "glyphicon glyphicon-user",
	 *          "dn": "uid=vagrant,ou=People,dc=srv,dc=world",
	 *          "id": "uid=vagrant",
	 *          "text": "vagrant",
	 *          "gidNumber": "1000"
	 *      }],
	 *      "status": true
	 *  }
	 * @throws IOException
	 */
	@RequestMapping(value="/getUserListTree")
	public @ResponseBody Map<String , Object> getUserListTree() throws IOException{
		return ldapService.getUserListTree();
	}
	
	/**
	 * 유저 상세정보를 불려온다
	 * @param userId 불려올 유저아이디
	 * @return
	 * @throws IOException
	 */
	@RequestMapping(value="/getUserInfo")
	public @ResponseBody Map<String , Object> getUserInfo(@RequestParam(value="id",required=false) String userId) throws IOException{
		Map<String, Object> result = new HashMap<String, Object>();
		result.put("status", true);
		result.put("result", ldapService.getUserInfo(userId));
		return result;
	}
	

	
	/**
	 * 유저를 추가하는 주소
	 * @param response
	 * @param sendData json 데이터을 받아온다
	 * 예시)
	 * [
	 * 		{
	 * 			name: 'uid',
	 * 			value: 'cend'
	 * 		},
	 * 		{
	 * 			name: 'loginShell',
	 * 			value: '/bin/bash'
	 * 		}
	 * ]
	 * @return
	 * @throws IOException
	 */
	@RequestMapping(value="/addUser",produces = "application/json",method=RequestMethod.POST)
	public @ResponseBody Map<String , Object> addUser(HttpServletResponse response,@RequestBody List<Map<String , Object>> sendData) throws IOException {
		Map<String, Object> result = ldapService.addUser(sendData);
		if(!(Boolean) result.get("status")){
			response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
		}
		return result;
	}
	
	/**
	 * 유저를 삭제하는 주소
	 * @param sendData json 데이터을 받아온다 (dn값만 필요)
	 * 예시)
	 * {
	 * 	dn: 'uid=vagrant,ou=People,dc=srv,dc=world'
	 * }
	 * @return
	 * @throws IOException
	 */
	@RequestMapping(value="/deleteUser",produces = "application/json",method=RequestMethod.POST)
	public @ResponseBody Map<String , Object> deleteUser(@RequestBody Map<String , Object> sendData) throws IOException {
		Map<String, Object> result = ldapService.deleteUser(sendData);
		return result;
	}
	
	
	
	/**
	 * ambari-view 에서 설정한 서버정보값을 불려온다
	 * @return Map<String , String>
	 */
	@RequestMapping(value="/serverInfo")
	public @ResponseBody Map<String , String> serverInfo(){
		Map<String, String> result = ldapService.serverInfo();
		return result;
	}
	
	/**
	 * LDAP Server 에서 uidNumber, gidNumber 값을 불려오는 메서드
	 * @return Map<String , Object>
	 */
	@RequestMapping(value="/getMaxUidNumber")
	public @ResponseBody Map<String , Object> getMaxUidNumber(){
		return ldapService.getMaxUidNumber();
	}

	/**
	 * 하둡 홈디렉토리 생성
	 * @param String 유저아이디
	 * @return 결과값 (JSON)
	 */
	@RequestMapping(value="/craetedHadoopHomeFolder")
	public @ResponseBody Map<String , Object> craetedHadoopHomeFolder(@RequestParam(value="id") String userId) {
		return ldapService.craetedHadoopHomeFolder(userId);
	}

}

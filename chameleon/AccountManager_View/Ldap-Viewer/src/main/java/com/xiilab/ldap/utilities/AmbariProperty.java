package com.xiilab.ldap.utilities;

import java.util.HashMap;
import java.util.Map;

import javax.servlet.ServletConfig;

import org.apache.ambari.view.ViewContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

@Component
public class AmbariProperty {
	
	@Autowired
	private ServletConfig servletConfig;
	
	
	/**
	 * Ambari LDAP View Setting 에서 설정한 환경설정정보를 읽어온다
	 * @return
	 */
	public Map<String,String> getServerConfigs() {
		Map<String,String> result = new HashMap<String,String>();
		
		//ViewContext viewContext = (ViewContext) request.getSession().getServletContext().getAttribute(ViewContext.CONTEXT_ATTRIBUTE);
		// ambari-view 에서 property 값을 읽어 올려면 ServletConfig 에서 읽어 와야함
				
			
		ViewContext viewContext = (ViewContext) servletConfig.getServletContext().getAttribute(ViewContext.CONTEXT_ATTRIBUTE);
		result.put("domain", viewContext.getProperties().get("SSH_User_ID")+"@"+viewContext.getProperties().get("SSH_URL"));
		result.put("sshPassWord", viewContext.getProperties().get("SSH_User_PASSWORD"));
		result.put("ManagerPassword", viewContext.getProperties().get("LDAP_MANAGER_PASSWORD"));
		result.put("dc", viewContext.getProperties().get("LDAP_DOMAIN_COMPONENT"));
		result.put("ManagerName", viewContext.getProperties().get("LDAP_MANAGER_NAME"));
		
		return result;
	}
	
	
}

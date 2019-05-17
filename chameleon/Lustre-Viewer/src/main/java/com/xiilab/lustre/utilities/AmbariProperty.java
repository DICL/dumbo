package com.xiilab.lustre.utilities;

import java.util.Map;

import javax.servlet.ServletConfig;

import org.apache.ambari.view.ViewContext;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;
import com.xiilab.lustre.model.AmbariViewConfigVO;

@Component
public class AmbariProperty {
	
	@Autowired
	private ServletConfig servletConfig;
	
	
	/**
	 * Ambari View Setting 에서 설정한 환경설정정보를 읽어온다
	 * @return
	 */
	public Map<String,String> getServerConfigs() {
		Map<String,String> result = null;
		//ViewContext viewContext = (ViewContext) request.getSession().getServletContext().getAttribute(ViewContext.CONTEXT_ATTRIBUTE);
		// ambari-view 에서 property 값을 읽어 올려면 ServletConfig 에서 읽어 와야함
		
		
		
		
		
		try {
			ViewContext viewContext = (ViewContext) servletConfig.getServletContext().getAttribute(ViewContext.CONTEXT_ATTRIBUTE);
			if(viewContext != null) {
				result = viewContext.getProperties();
			}
		} catch (Exception e) {
			//e.printStackTrace();
		}
		
		return result;
	}
	
	
	/**
	 * server property 을 읽어와서 AmbariViewConfigVO 적재하는 메서드
	 * @return
	 */
	public AmbariViewConfigVO readAmbariViewConfigs() {
		AmbariViewConfigVO result = new AmbariViewConfigVO();
		Map<String,String> tmp = getServerConfigs();
		
		result.setAmbari_containername(tmp.get("ambari.server.containername"));
		result.setAmbari_url(tmp.get("ambari.server.url"));
		result.setAmbari_username(tmp.get("ambari.server.username"));
		result.setAmbari_password(tmp.get("ambari.server.password"));
		
		return result;
	}
	
	
}

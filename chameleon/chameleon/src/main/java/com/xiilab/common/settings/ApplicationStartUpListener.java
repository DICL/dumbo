package com.xiilab.common.settings;


import java.io.IOException;

import javax.servlet.ServletContext;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import com.xiilab.ambari.AmbariStatusDAO;

/**
 * 서블릿 이벤트 처리 클래스
 * @author xiilab
 *
 */
@Component
public class ApplicationStartUpListener {
 
	static final Logger LOGGER = LoggerFactory.getLogger(ApplicationStartUpListener.class);
	
	@Autowired
    private ServletContext servletContext;
	
	@Autowired
	private AmbariStatusDAO statusDAO;
	
	/**
	 * spring boot 시작시 동작하는 이벤트
	 */
	@EventListener(ApplicationReadyEvent.class)
	public void doSomethingAfterStartup() {
		LOGGER.info("Started up Chameleon And Get Ambari setting .....");
		
		// ambari host name & clustre name 가져오기
		String ambari_host = statusDAO.getAmbariServerHostName();
		String cluster_name = "supercom_test";
		
		servletContext.setAttribute("ambari_host", ambari_host);
		try {
			LOGGER.info("find ambari host name : [{}]",ambari_host);
			cluster_name = statusDAO.getAmbariServerClustreName(ambari_host);
			servletContext.setAttribute("ambari_cluster_name", cluster_name);
			LOGGER.info("find ambari clustre name : [{}]",cluster_name);
			
			String resourceManagerHost = statusDAO.getResourceManagerHostName(ambari_host, cluster_name);
			servletContext.setAttribute("yarn_resource_manager_host", resourceManagerHost);
			LOGGER.info("find yarn resource manager host : [{}]",resourceManagerHost);
			
			String metrics_colletor_host = statusDAO.getAmbariMetricCollectorHost(ambari_host, cluster_name);
			servletContext.setAttribute("metrics_colletor_host", metrics_colletor_host);
			LOGGER.info("find ambari mectric colletor host : [{}]",metrics_colletor_host);
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			//servletContext.setAttribute("ambari_cluster_name", cluster_name);
		}
	}
}

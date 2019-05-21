package com.xiilab.websocket;

import java.util.Map;

import javax.servlet.http.HttpSession;

import org.springframework.http.server.ServerHttpRequest;
import org.springframework.http.server.ServerHttpResponse;
import org.springframework.http.server.ServletServerHttpRequest;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.server.support.HttpSessionHandshakeInterceptor;

//import com.xiilab.models.ServerInfomationVO;

public class TerminalExecHandshakeInterceptor extends HttpSessionHandshakeInterceptor{
	
	@Override
	// ws://192.168.1.40:8084/terminal?host=node01
	public boolean beforeHandshake(ServerHttpRequest request, ServerHttpResponse response, WebSocketHandler wsHandler,
			Map<String, Object> attributes) throws Exception {
		
		// https://stackoverflow.com/questions/42243543/how-to-get-session-id-in-spring-websocketstompclient
		if (request instanceof ServletServerHttpRequest) {
            ServletServerHttpRequest servletRequest = (ServletServerHttpRequest) request;
            HttpSession session = servletRequest.getServletRequest().getSession();
           
//            Map<String,ServerInfomationVO> serverInfo = (Map<String, ServerInfomationVO>) session.getAttribute("serverInfo");
//            if(serverInfo != null) {
//            	attributes.put("serverInfo", serverInfo);
//            }else {
//            	
//            }
            
            String host_name = servletRequest.getServletRequest().getParameter("host");
            if(host_name != null) {
            	attributes.put("host_name", host_name);
            }else {
            	
            }
        }
		
		return super.beforeHandshake(request, response, wsHandler, attributes);
	}
	
	
	@Override
	public void afterHandshake(ServerHttpRequest request, ServerHttpResponse response, WebSocketHandler wsHandler,
			Exception ex) {
		super.afterHandshake(request, response, wsHandler, ex);
	}
}

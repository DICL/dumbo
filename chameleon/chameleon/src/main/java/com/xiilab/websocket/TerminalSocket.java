package com.xiilab.websocket;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;


public class TerminalSocket extends TextWebSocketHandler {
	
	private final TerminalService terminalService;
	
	@Autowired
	public TerminalSocket(TerminalService terminalService) {
		this.terminalService = terminalService;
	}
	
	/* 
	 * 웹소켓 서버 시작시 동작하는 메서드
	 * (non-Javadoc)
	 * @see org.springframework.web.socket.handler.AbstractWebSocketHandler#afterConnectionEstablished(org.springframework.web.socket.WebSocketSession)
	 */
	@Override
	public void afterConnectionEstablished(WebSocketSession session) throws Exception {
		
		
		
		// 세션 입력
		terminalService.setWebSocketSession(session);
		super.afterConnectionEstablished(session);
	}
	
	
	
	/* 
	 * 클라이언트가 서버로 메세지 수신할경우 동작하는 메서드
	 * (non-Javadoc)
	 * @see org.springframework.web.socket.handler.AbstractWebSocketHandler#handleTextMessage(org.springframework.web.socket.WebSocketSession, org.springframework.web.socket.TextMessage)
	 */
	@Override
	protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
		
		// 메세지 형식 {type: "TERMINAL_COMMAND", command: ""} (json) 을  Map 으로 변경
		Map<String, String> messageMap = getMessageMap(message);
		
		// 메세제 타입에 따른 분류처리
        if (messageMap.containsKey("type")) {
            String type = messageMap.get("type");

            switch (type) {
            	
            	// TERMINAL_AUTH : server 에 유저 정보 입력
            	case "TERMINAL_AUTH":
            		terminalService.setUserName(messageMap.get("user_name"));
            		terminalService.setUserPassWord(messageMap.get("user_pass"));
            		break;
            	
            	// TERMINAL_INIT & TERMINAL_READY & setup
                // SSH 세션 연결및 통신을 위한 쓰레드 생성
                case "TERMINAL_INIT": case "TERMINAL_READY": case "setup":
                	if(terminalService.getUserName() != null && terminalService.getUserPassWord() != null && !"".equals(terminalService.getUserName()) && !"".equals(terminalService.getUserPassWord())) {
                		terminalService.onTerminalInit();
                	}else {
                		throw new SecurityException("Authentication FAIL");
                	}
                    break;
                // TERMINAL_COMMAND : 받은 메세지를 해당 쓰레드에 내부에 있는 스트림에 메세지 입력
                case "TERMINAL_COMMAND": case "stdin": 
                    terminalService.onCommand(messageMap.get("command"));
                    break;
               // 터미널 리사이즈
                case "TERMINAL_RESIZE": case "set_size":
                    terminalService.onTerminalResize(messageMap.get("columns"), messageMap.get("rows"), messageMap.get("width"), messageMap.get("height"));
                    break;
                default:
                    throw new RuntimeException("Unrecodnized action");
            }
        }
	}
	
	
	private Map<String, String> getMessageMap(TextMessage message) {
        try {
            Map<String, String> map = new ObjectMapper().readValue(message.getPayload(), new TypeReference<Map<String, String>>() {
            });

            return map;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new HashMap<>();
    }
	
	
	
	/* (non-Javadoc)
	 * @see org.springframework.web.socket.handler.AbstractWebSocketHandler#afterConnectionClosed(org.springframework.web.socket.WebSocketSession, org.springframework.web.socket.CloseStatus)
	 */
	@Override
	public void afterConnectionClosed(WebSocketSession session, CloseStatus status) throws Exception {
		terminalService.disConnect();
		super.afterConnectionClosed(session, status);
	}
	
	
	
	/* (non-Javadoc)
	 * @see org.springframework.web.socket.handler.AbstractWebSocketHandler#handleTransportError(org.springframework.web.socket.WebSocketSession, java.lang.Throwable)
	 */
	@Override
	public void handleTransportError(WebSocketSession session, Throwable exception) throws Exception {
		super.handleTransportError(session, exception);
	}
	
	/* (non-Javadoc)
	 * @see org.springframework.web.socket.handler.AbstractWebSocketHandler#supportsPartialMessages()
	 */
	@Override
    public boolean supportsPartialMessages() {
        return super.supportsPartialMessages();
    }
	
	
}

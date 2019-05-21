package com.xiilab.common.settings;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.socket.WebSocketHandler;
import org.springframework.web.socket.config.annotation.EnableWebSocket;
import org.springframework.web.socket.config.annotation.WebSocketConfigurer;
import org.springframework.web.socket.config.annotation.WebSocketHandlerRegistry;
import org.springframework.web.socket.handler.PerConnectionWebSocketHandler;

import com.xiilab.websocket.TerminalExecHandshakeInterceptor;
import com.xiilab.websocket.TerminalSocket;





/**
 * 웹소켓 환경설정 / 빈등록 / 웹소켓 맵핑,인터럽트 설정
 * @author xiilab
 *
 */
@Configuration
@EnableWebSocket
public class WebSocketConfig implements WebSocketConfigurer {

	@Override
    public void registerWebSocketHandlers(WebSocketHandlerRegistry registry) {
        registry
                .addHandler(terminalSocket(), "/terminal").setAllowedOrigins("*").addInterceptors(new TerminalExecHandshakeInterceptor());
    }

    @Bean
    public WebSocketHandler terminalSocket() {
        WebSocketHandler webSocketHandler = new PerConnectionWebSocketHandler(TerminalSocket.class);
        return webSocketHandler;
    }
	
	
}

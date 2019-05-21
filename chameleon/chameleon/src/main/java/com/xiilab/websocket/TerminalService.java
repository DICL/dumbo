package com.xiilab.websocket;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;

import org.springframework.context.annotation.Scope;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jcraft.jsch.ChannelShell;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.JSchException;
import com.jcraft.jsch.Session;



@Component
@Scope("prototype")
public class TerminalService {

	private boolean isReady;
    private String[] termCommand;
    
    private Integer columns = 20;
    private Integer rows = 10;
    private Integer width = columns * 12;
    private Integer height = columns * 16;
    
   
    private BufferedReader inputReader;
    private BufferedReader errorReader;
    private BufferedWriter outputWriter;
    private WebSocketSession webSocketSession;
    
    private String userName;
    private String userPassWord;
    
    private Session session;
    private ChannelShell channel;
	
    private LinkedBlockingQueue<String> commandQueue = new LinkedBlockingQueue<>();
	
	/**
	 * 웹 소켓 생성시 ssh 세션 연결및 쓰레드에 적재
	 */
	public void onTerminalInit() {
		ThreadHelper.start(() -> {
            isReady = true;
            try {
                initializeProcess();
            } catch (com.jcraft.jsch.JSchException e) {
            	try {
					sendAlert("Invalid login or password.");
				} catch (IOException e2) {
					e2.printStackTrace();
				}
            	//throw new SecurityException(e.toString());
            } catch (Exception e1) {
                 e1.printStackTrace();
            }
        });
	}

	/**
	 * ssh 세션 생성
	 * @throws Exception
	 */
	private void initializeProcess() throws Exception {
		
		JSch jsch = new JSch();
		
		// 기본정보
		int port = 22;
//		String host = "192.168.1.191";
//		String userId = "root";
//		String password = "vagrant";
		
		String host = (String) webSocketSession.getAttributes().get("host_name");
		String userId = this.userName;
		String password = this.userPassWord;
		
		
		// jsch 세션 생성 
		session = jsch.getSession(userId, host ,port);
		if (session == null) {
			throw new Exception("session is null");
		}
		// 패스워드 입력
		session.setPassword(password);
		// 호스트 정보를 받지 않음
		session.setConfig("StrictHostKeyChecking", "no");
//		session.connect(30000);
		session.connect();
		
		
		// 입력 및 출력 스트림처리를 위한 채널생성
		channel = (ChannelShell) session.openChannel("shell");
		
		channel.setAgentForwarding(true);
		// channel.setPtyType("vt102");
		// 터미널 사이즈 조정
		channel.setPtySize(this.columns, this.rows, this.width, this.height);
//		channel.connect(1000);
		
		// 쉘 접속
		channel.connect();
		
		// 입력스트림을 저장
		this.inputReader = new BufferedReader(new InputStreamReader(channel.getInputStream()));
        //this.errorReader = new BufferedReader(new InputStreamReader(channel.getExtInputStream() ));
		
		// 출력 스트림 저장
        this.outputWriter = new BufferedWriter(new OutputStreamWriter(channel.getOutputStream()));
		
        ThreadHelper.start(() -> {
            printReader(inputReader);
        });

        ThreadHelper.start(() -> {
            printReader(errorReader);
        });
		
//        channel.wait();
	}

	/**
	 * ssh 에서 받은 스트림을 문자열로 변환
	 * @param bufferedReader
	 */
	private void printReader(BufferedReader bufferedReader) {
		try {
            int nRead;
            char[] data = new char[10 * 1024];
            if(bufferedReader != null) {
            	while ((nRead = bufferedReader.read(data, 0, data.length)) != -1) {
            		StringBuilder builder = new StringBuilder(nRead);
            		builder.append(data, 0, nRead);
            		print(builder.toString());
            	}
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
	}

	/**
	 * 클라이언트에게 문자 전송
	 * @param text
	 * @throws IOException
	 */
	private void print(String text)  throws IOException {
		Map<String, String> map = new HashMap<>();
        map.put("type", "TERMINAL_PRINT");
        map.put("text", text);

        String message = new ObjectMapper().writeValueAsString(map);

        webSocketSession.sendMessage(new TextMessage(message));
	}
	
	
	/**
	 * 클라이언트에게 경고문 전송
	 * @param text
	 * @throws IOException
	 */
	private void sendAlert(String text) throws IOException {
		Map<String, String> map = new HashMap<>();
        map.put("type", "TERMINAL_ALERT");
        map.put("text", text);

        String message = new ObjectMapper().writeValueAsString(map);

        webSocketSession.sendMessage(new TextMessage(message));
	}
	

	/**
	 * 클라이언트에서 받은 문자열을 스트림에 적재
	 * @param command
	 * @throws InterruptedException
	 */
	public void onCommand(String command ) throws InterruptedException {
//		System.out.println(command + " -onCommand");
		 if (Objects.isNull(command)) {
	            return;
	        }

	        commandQueue.put(command);
	        ThreadHelper.start(() -> {
	            try {
	                outputWriter.write(commandQueue.poll());
	                outputWriter.flush();
	            } catch (IOException e) {
	                e.printStackTrace();
	            }
	        });
	}

	/**
	 * 터미널 사이즈 조정
	 * @param columns
	 * @param rows
	 */
	public void onTerminalResize(String columns, String rows , String width , String height) {
		if (Objects.nonNull(columns) && Objects.nonNull(rows)) {
			this.columns = Integer.valueOf(columns);
            this.rows = Integer.valueOf(rows);
            this.width = Integer.valueOf(rows);
            this.height = Integer.valueOf(rows);
            
            if (Objects.nonNull(channel)) {
            	channel.setPtySize(this.columns, this.rows, this.width, this.height);
            }
		}
	}

	public void setWebSocketSession(WebSocketSession webSocketSession) {
		this.webSocketSession = webSocketSession;
	}
	
	
	public WebSocketSession getWebSocketSession() {
        return webSocketSession;
    }

	/**
	 * ssh 통신 종료
	 */
	public void disConnect() {
		if (Objects.nonNull(channel)) {
			this.channel.disconnect();
		}
		
		if (Objects.nonNull(session)) {
			this.session.disconnect();
		}
	}

	public String getUserName() {
		return userName;
	}

	public void setUserName(String userName) {
		this.userName = userName;
	}

	public String getUserPassWord() {
		return userPassWord;
	}

	public void setUserPassWord(String userPassWord) {
		this.userPassWord = userPassWord;
	}

}

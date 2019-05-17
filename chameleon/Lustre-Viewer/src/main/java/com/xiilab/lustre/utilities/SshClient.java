package com.xiilab.lustre.utilities;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import com.jcraft.jsch.Channel;
import com.jcraft.jsch.ChannelExec;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.Session;
import com.xiilab.lustre.api.LustreDAO;
import com.xiilab.lustre.model.LustreNodesVO;

@Component
public class SshClient {
	
	static final Logger LOGGER = LoggerFactory.getLogger(SshClient.class);
	
	@Autowired
	private LustreDAO lustreDAO;
	
	
	/**
	 * ssh 명령어 전송
	 * @param lustre_info 호스트정보
	 * @param command 명령어
	 * @param save_log_rowKey 유니크 키
	 * @param log_label 명령어 이름
	 * @return
	 */
	public Map<String, Object> execjschForLogging(LustreNodesVO lustre_info,String command,String save_log_rowKey, String log_label){
		// 1. JSch 객체를 생성한다.
		JSch jsch =new JSch();
				
		Session session = null;
		Channel channel = null;
				
		String username = null;
		String host = null;
		String password = null;
		int port = 0;
				
		int SSH_TIME_OUT_MILLISECOND = 60000;
				
		String private_key_path = System.getProperty("user.home") + "/" + lustre_info.getPrivate_key();
			
		Map<String, Object> resultMap = new HashMap<>();
		
		try {
			username = lustre_info.getUser_id();
			host = lustre_info.getHost_name();
			password = lustre_info.getPassword();
			port = lustre_info.getSsh_port();
			
			if(lustre_info.getPrivate_key() != null && !lustre_info.getPrivate_key().equals("") ) {
				jsch.addIdentity(private_key_path);
			}
			
			// 2. 세션 객체를 생성한다 (사용자 이름, 접속할 호스트, 포트를 인자로 준다.)
			session = jsch.getSession(username, host, port);
			
			if(lustre_info.getPrivate_key() == null || lustre_info.getPrivate_key().equals("") ) {
				// 3. 패스워드를 설정한다.
				session.setPassword(password);
			}
			
			
			// 4. 세션과 관련된 정보를 설정한다.
			java.util.Properties config = new java.util.Properties();
			// 4-1. 호스트 정보를 검사하지 않는다.
			config.put("StrictHostKeyChecking", "no");
			session.setConfig(config);
			
			// 5. 접속한다.
			session.connect();
			
			// 6. sftp 채널을 연다.
			channel = session.openChannel("exec");
			// 7. 채널을 SSH용 채널 객체로 캐스팅한다
			ChannelExec channelExec = ((ChannelExec) channel);
				
			channelExec.setCommand(command);
			channel.setInputStream(null);
			
			// 8. 채널에 아웃풋설정을 한다.
			InputStream in = channel.getInputStream(); // channel.getInputStream();
			// error stream 추가
			final InputStream err = channelExec.getErrStream();// <- 일반 에러 스트림
			
			channelExec.connect(SSH_TIME_OUT_MILLISECOND);
			
			saveCommandLog(save_log_rowKey,command,log_label,"command",lustre_info.getHost_name());
			resultMap = getChannelOutputSaveLog(channel, in, err, save_log_rowKey,log_label,lustre_info.getHost_name());
			
		}catch (Exception e) {
			e.printStackTrace();
			LOGGER.error(e.getMessage());
			
			// 실패시 로그와 결과값 출력
			resultMap.put("status", false);
			resultMap.put("log", e.toString());
			return resultMap;
			
		} finally {
            if (channel != null) {
                channel.disconnect();
            }
            if (session != null) {
                session.disconnect();
            }
        }
		
		
		
		//resultMap.put("status", true);
		//resultMap.put("log", result);
		
		
		return resultMap;
	}
	
	
	
	/**
	 * ssh 결과를 디비에 저장하는 메서드
	 * @param channel
	 * @param in
	 * @param err
	 * @param save_log_rowKey
	 * @param log_label
	 * @param hostname
	 * @return
	 * @throws IOException
	 */
	private Map<String, Object> getChannelOutputSaveLog(Channel channel, InputStream in, InputStream err, String save_log_rowKey, String log_label, String hostname) throws IOException {
		StringBuffer result = new StringBuffer();
		Map<String, Object> resultMap = new HashMap<>();
		resultMap.put("status", true);
		
        String line = "";
        while (true){
        	while (in.available() > 0) {
        		BufferedReader info_reader = new BufferedReader(new InputStreamReader(in, "UTF-8"));
        		while(info_reader.ready())  {
        			line = info_reader.readLine();
        			result.append(line+"\n");
        			saveCommandLog(save_log_rowKey,line,log_label,"info",hostname);
            	}
        	}
        	
        	while (err.available() > 0) {
        		BufferedReader err_reader = new BufferedReader(new InputStreamReader(err, "UTF-8"));
        		while(err_reader.ready()) {
        			line = err_reader.readLine();
        			result.append(line+"\n");
        			saveCommandLog(save_log_rowKey,line,log_label,"error",hostname);
        		}
        		resultMap.put("status", false);
        	}
        	
        	if(line.contains("logout")){
                break;
            }

            if (channel.isClosed()){
                break;
            }
            try {
                Thread.sleep(100);
            } catch (Exception ee){}
        }
		
        resultMap.put("log", result.toString());
		
		return resultMap;
	}



	/**
	 * 로그 저장
	 * @param save_log_rowKey 유니크 키값
	 * @param line 로그 내용
	 * @param log_label 로그 이름
	 * @param type 로그 타입
	 * @param hostname 호스트이름
	 */
	private void saveCommandLog(String save_log_rowKey, String line, String log_label, String type, String hostname) {
		lustreDAO.saveCommandLog(save_log_rowKey,line,log_label,type,hostname);
	}



	/**
	 * ssh 명령어 전송
	 * @param lustre_info 호스트정보
	 * @param command 명령어
	 * @param save_log 로그저장여부
	 * @return
	 */
	public Map<String, Object> execjsch(LustreNodesVO lustre_info,String command){
		// 1. JSch 객체를 생성한다.
		JSch jsch =new JSch();
		String result =null;
		
		Session session = null;
		Channel channel = null;
		
		String username = null;
		String host = null;
		String password = null;
		int port = 0;
		
		int SSH_TIME_OUT_MILLISECOND = 60000;
		
		String private_key_path = System.getProperty("user.home") + "/" + lustre_info.getPrivate_key();
		
		Map<String, Object> resultMap = new HashMap<>();
		try {
			username = lustre_info.getUser_id();
			host = lustre_info.getHost_name();
			password = lustre_info.getPassword();
			port = lustre_info.getSsh_port();
			
			if(lustre_info.getPrivate_key() != null && !lustre_info.getPrivate_key().equals("") ) {
				jsch.addIdentity(private_key_path);
			}
			
			// 2. 세션 객체를 생성한다 (사용자 이름, 접속할 호스트, 포트를 인자로 준다.)
			session = jsch.getSession(username, host, port);
			
			if(lustre_info.getPrivate_key() == null || lustre_info.getPrivate_key().equals("") ) {
				// 3. 패스워드를 설정한다.
				session.setPassword(password);
			}
			
			
			// 4. 세션과 관련된 정보를 설정한다.
			java.util.Properties config = new java.util.Properties();
			// 4-1. 호스트 정보를 검사하지 않는다.
			config.put("StrictHostKeyChecking", "no");
			session.setConfig(config);
			
			// 5. 접속한다.
			session.connect();
			
			// 6. sftp 채널을 연다.
			channel = session.openChannel("exec");
			// 7. 채널을 SSH용 채널 객체로 캐스팅한다
			ChannelExec channelExec = ((ChannelExec) channel);
				
			channelExec.setCommand(command);
			channel.setInputStream(null);
			
			// 8. 채널에 아웃풋설정을 한다.
			InputStream in = channel.getInputStream(); // channel.getInputStream();
			// error stream 추가
			final InputStream err = channelExec.getErrStream();// <- 일반 에러 스트림
			
			channelExec.connect(SSH_TIME_OUT_MILLISECOND);
			
			result = getChannelOutput(channel, in, err);
			
		}catch (Exception e) {
			e.printStackTrace();
			LOGGER.error(e.getMessage());
			
			// 실패시 로그와 결과값 출력
			resultMap.put("status", false);
			resultMap.put("log", e.toString());
			return resultMap;
			
		} finally {
            if (channel != null) {
                channel.disconnect();
            }
            if (session != null) {
                session.disconnect();
            }
        }
		
		
		
		resultMap.put("status", true);
		resultMap.put("log", result);
		return resultMap;
	}

	/**
	 * 명령어 전달결과를 텍스트로 출력
	 * @param channel
	 * @param in
	 * @param err 
	 * @return
	 * @throws IOException
	 */
	private String getChannelOutput(Channel channel, InputStream in, InputStream err) throws IOException {
		byte[] buffer = new byte[1024];
        StringBuilder strBuilder = new StringBuilder();

        String line = "";
        while (true){
            while (in.available() > 0) {
                int i = in.read(buffer, 0, 1024);
                if (i < 0) {
                    break;
                }
                strBuilder.append(new String(buffer, 0, i));
//                System.out.println(line);
            }
            
            while (err.available() > 0) {
                int i = err.read(buffer, 0, 1024);
                if (i < 0) {
                    break;
                }
                strBuilder.append(new String(buffer, 0, i));
            }

            if(line.contains("logout")){
                break;
            }

            if (channel.isClosed()){
                break;
            }
            try {
                Thread.sleep(100);
            } catch (Exception ee){}
        }

        return strBuilder.toString();  
	}
	
}

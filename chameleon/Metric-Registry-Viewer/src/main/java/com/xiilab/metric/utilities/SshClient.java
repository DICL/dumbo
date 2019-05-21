package com.xiilab.metric.utilities;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import com.jcraft.jsch.Channel;
import com.jcraft.jsch.ChannelExec;
import com.jcraft.jsch.ChannelSftp;
import com.jcraft.jsch.JSch;
import com.jcraft.jsch.Session;
import com.jcraft.jsch.SftpException;
import com.xiilab.metric.model.YarnAppNodesVO;

@Component
public class SshClient {
	static final Logger LOGGER = LoggerFactory.getLogger(SshClient.class);
	
	
	
	
	/**
	 * sftp 파일전송
	 * 
	 * @param serverVO 서버정보
	 * @param remoteDir 전송위치
	 * @param file 전송파일
	 * @return
	 */
	public Map<String, Object> sftpSend(YarnAppNodesVO yarnAppNodes, String remoteDir , File file) {
		Map<String, Object> result = new HashMap<>();
		Session session = null;
		Channel channel = null;
		
		String username = null;
		String host = null;
		String id_rsa = null;
		String password = null;
		int port = 0;
		
		FileInputStream in = null;
		
		
		try {
			username = yarnAppNodes.getUser_id();
			host = yarnAppNodes.getHost_name();
			port = yarnAppNodes.getSsh_port();
			id_rsa = yarnAppNodes.getPrivate_key();
			password = yarnAppNodes.getPassword();
		} catch (Exception e) {
			// 실패시 로그와 결과값 출력
			LOGGER.error(e.getMessage());
			result.put("status", false);
			result.put("log", e.toString());
			return result;
		}
		
		
		// 1. JSch 객체를 생성한다.
		JSch jsch = new JSch();
		
		try {
			if(yarnAppNodes.getPrivate_key() != null && !yarnAppNodes.getPrivate_key().equals("") ) {
				jsch.addIdentity(id_rsa);
			}
			// 2. 세션 객체를 생성한다(사용자 이름, 접속할 호스트, 포트를 인자로 전달한다.)
			session = jsch.getSession(username, host, port);
			// 3. 패스워드를 설정한다.
			if(password != null ) {
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
			channel = session.openChannel("sftp");
			 
			// 7. 채널에 연결한다.
			channel.connect();
			
			// 8. 채널을 FTP용 채널 객체로 캐스팅한다.
			ChannelSftp channelSftp = (ChannelSftp) channel;
			
			// 8.1 해당 유저의 홈디렉토리
			String currentDirectory=channelSftp.pwd();
			
			// 8.2  '~/' -> /home/<username>/  
			String destFolder = remoteDir.replaceFirst("^~",currentDirectory);
			
			// 8.3 폴더가 없으면 자동 생성
			// 업로드하려는 위치르 디렉토리를 변경한다.
			// 하위폴더까지 생성
			try {
				channelSftp.cd(destFolder);
			}
			catch ( SftpException e ) {
				String[] complPath = destFolder.split("/");
				channelSftp.cd("/");
				for (String dir : complPath) {
	                if (dir.length() > 0) {
	                    try {
	                    	channelSftp.cd(dir);
	                    } catch (SftpException e2) {
	                    	channelSftp.mkdir(dir);
	                    	channelSftp.cd(dir);
	                    }
	                }
	            }
				channelSftp.cd( destFolder );
	        }
			
			// 입력 파일을 가져온다.
	        in = new FileInputStream(file);
	        
	        // 파일을 업로드한다.
	        channelSftp.put(in, file.getName());
		
		} catch (Exception e) {
			// 실패시 로그와 결과값 출력
			LOGGER.error(e.getMessage());
			result.put("status", false);
			result.put("log", e.toString());
			return result;
		} finally {
            if (session != null) {
                session.disconnect();
            }
        }
		
		
		
		result.put("status", true);
		result.put("log", "");
		return result;
	}
	
	
	/**
	 * ssh 명령어 전송
	 * @param lustre_info 호스트정보
	 * @param command 명령어
	 * @param save_log 로그저장여부
	 * @return
	 */
	public Map<String, Object> execjsch(YarnAppNodesVO yarnAppNodes,String command){
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
		
		//String private_key_path = System.getProperty("user.home") + "/" + yarnAppNodes.getPrivate_key();
		String private_key_path = yarnAppNodes.getPrivate_key();
		Map<String, Object> resultMap = new HashMap<>();
		try {
			username = yarnAppNodes.getUser_id();
			host = yarnAppNodes.getHost_name();
			password = yarnAppNodes.getPassword();
			port = yarnAppNodes.getSsh_port();
			
			if(yarnAppNodes.getPrivate_key() != null && !yarnAppNodes.getPrivate_key().equals("") ) {
				jsch.addIdentity(private_key_path);
			}
			
			// 2. 세션 객체를 생성한다 (사용자 이름, 접속할 호스트, 포트를 인자로 준다.)
			session = jsch.getSession(username, host, port);
			
			if(yarnAppNodes.getPrivate_key() == null || yarnAppNodes.getPrivate_key().equals("") ) {
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

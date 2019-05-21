package com.xiilab.metric.model;

public class YarnAppNodesVO {
	private Long num;
    private String host_name; 
    private Integer ssh_port;
    private String user_id;
    private String password;
    private String private_key;
    
    
	
	
	
	public String getHost_name() {
		return host_name;
	}
	public void setHost_name(String host_name) {
		this.host_name = host_name;
	}
	public String getUser_id() {
		return user_id;
	}
	public void setUser_id(String user_id) {
		this.user_id = user_id;
	}
	public String getPrivate_key() {
		return private_key;
	}
	public void setPrivate_key(String private_key) {
		this.private_key = private_key;
	}
	public Long getNum() {
		return num;
	}
	public void setNum(Long num) {
		this.num = num;
	}
	public String getPassword() {
		return password;
	}
	public void setPassword(String password) {
		this.password = password;
	}
	public Integer getSsh_port() {
		return ssh_port;
	}
	public void setSsh_port(Integer ssh_port) {
		this.ssh_port = ssh_port;
	}
	
}

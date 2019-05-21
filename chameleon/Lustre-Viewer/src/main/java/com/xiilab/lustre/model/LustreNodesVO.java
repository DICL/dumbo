package com.xiilab.lustre.model;

import java.util.List;

public class LustreNodesVO {
	private Long num;
    private String node_type; 
    private Integer index; 
    private String host_name; 
    private Integer ssh_port;
    private String user_id;
    private String password;
    private String private_key;
    
    private String network_device;
    private String network_option;
    
    private String lustre_client_folder;
    
    private List<LustreNodesVO> list;
    private List<DiskInforVO> disk_list;
    
    private Long file_system_num;
    // 검색용 임시 겍체
    private Integer fs_step;
    
	
	
	
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
	public String getNetwork_device() {
		return network_device;
	}
	public void setNetwork_device(String network_device) {
		this.network_device = network_device;
	}
	public List<DiskInforVO> getDisk_list() {
		return disk_list;
	}
	public void setDisk_list(List<DiskInforVO> disk_list) {
		this.disk_list = disk_list;
	}
	public List<LustreNodesVO> getList() {
		return list;
	}
	public void setList(List<LustreNodesVO> list) {
		this.list = list;
	}
	public Integer getIndex() {
		return index;
	}
	public void setIndex(Integer index) {
		this.index = index;
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
	public String getNode_type() {
		return node_type;
	}
	public void setNode_type(String node_type) {
		this.node_type = node_type;
	}
	public String getLustre_client_folder() {
		return lustre_client_folder;
	}
	public void setLustre_client_folder(String lustre_client_folder) {
		this.lustre_client_folder = lustre_client_folder;
	}
	public String getNetwork_option() {
		return network_option;
	}
	public void setNetwork_option(String network_option) {
		this.network_option = network_option;
	}
	public Long getFile_system_num() {
		return file_system_num;
	}
	public void setFile_system_num(Long file_system_num) {
		this.file_system_num = file_system_num;
	}
	public Integer getFs_step() {
		return fs_step;
	}
	public void setFs_step(Integer fs_step) {
		this.fs_step = fs_step;
	}
	
}

package com.xiilab.lustre.model;

import java.util.List;

public class DiskInforVO {
	private Long num; // 기본키
    private String disk_type; // MDT, OST
    private Integer index; // MDT or OST index
    private String disk_name; // /dev/sda1, /dev/sdb1 ...
    private Long lustre_nodes_key; // lustre_nodes private key
	
    private Boolean is_activate; // true: activate disk false: deactivate disk
    private Boolean is_remove; // true 디스크 존재 , false : 디스크 삭제
    
    private Boolean remove_disk_frag; // true 디스크 삭제동작 (미사용)
    
    private String host_name; // 검색을 위한 호스트네임(임시)
    private String backup_file_localtion; // 백업을 위한 백업파일 저장위치
    
    private String disk_size; // 디스크 사이즈
    
    private Long file_system_num;
    
	
	public String getDisk_name() {
		return disk_name;
	}
	public void setDisk_name(String disk_name) {
		this.disk_name = disk_name;
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
	public Long getLustre_nodes_key() {
		return lustre_nodes_key;
	}
	public void setLustre_nodes_key(Long lustre_nodes_key) {
		this.lustre_nodes_key = lustre_nodes_key;
	}
	public String getDisk_type() {
		return disk_type;
	}
	public void setDisk_type(String disk_type) {
		this.disk_type = disk_type;
	}
	public Boolean getIs_activate() {
		return is_activate;
	}
	public void setIs_activate(Boolean is_activate) {
		this.is_activate = is_activate;
	}
	public Boolean getRemove_disk_frag() {
		return remove_disk_frag;
	}
	public void setRemove_disk_frag(Boolean remove_disk_frag) {
		this.remove_disk_frag = remove_disk_frag;
	}
	public String getHost_name() {
		return host_name;
	}
	public void setHost_name(String host_name) {
		this.host_name = host_name;
	}
	public String getBackup_file_localtion() {
		return backup_file_localtion;
	}
	public void setBackup_file_localtion(String backup_file_localtion) {
		this.backup_file_localtion = backup_file_localtion;
	}
	public Long getFile_system_num() {
		return file_system_num;
	}
	public void setFile_system_num(Long file_system_num) {
		this.file_system_num = file_system_num;
	}
	public Boolean getIs_remove() {
		return is_remove;
	}
	public void setIs_remove(Boolean is_remove) {
		this.is_remove = is_remove;
	}
	public String getDisk_size() {
		return disk_size;
	}
	public void setDisk_size(String disk_size) {
		this.disk_size = disk_size;
	}
}

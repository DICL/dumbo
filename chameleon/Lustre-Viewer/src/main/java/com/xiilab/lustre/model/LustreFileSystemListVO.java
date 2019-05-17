package com.xiilab.lustre.model;

public class LustreFileSystemListVO {
	private Long num;
	private String fs_name;
	private String origin_fs_name;
	private Integer fs_step;
	
	private Boolean is_client_mount; // 마운트여부 확인
	private String lustre_client_folder; // 클라이언트 폴더
	
	private Boolean is_remove; // 삭제여부
	
	
	public Long getNum() {
		return num;
	}
	public void setNum(Long num) {
		this.num = num;
	}
	public String getFs_name() {
		return fs_name;
	}
	public void setFs_name(String fs_name) {
		this.fs_name = fs_name;
	}
	public Integer getFs_step() {
		return fs_step;
	}
	public void setFs_step(Integer fs_step) {
		this.fs_step = fs_step;
	}
	public String getOrigin_fs_name() {
		return origin_fs_name;
	}
	public void setOrigin_fs_name(String origin_fs_name) {
		this.origin_fs_name = origin_fs_name;
	}
	public Boolean getIs_client_mount() {
		return is_client_mount;
	}
	public void setIs_client_mount(Boolean is_client_mount) {
		this.is_client_mount = is_client_mount;
	}
	public String getLustre_client_folder() {
		return lustre_client_folder;
	}
	public void setLustre_client_folder(String lustre_client_folder) {
		this.lustre_client_folder = lustre_client_folder;
	}
	public Boolean getIs_remove() {
		return is_remove;
	}
	public void setIs_remove(Boolean is_remove) {
		this.is_remove = is_remove;
	}
}

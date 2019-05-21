package com.xiilab.metric.model;

public class AmbariYarnAppMonitorConfig {
	
	private String timescaleDB_database ;
	private String timescaleDB_url ;
	private String timescaleDB_port ;
	private String timescaleDB_password ;
	private String timescaleDB_username ;
	
	
	public String getTimescaleDB_database() {
		return timescaleDB_database;
	}
	public void setTimescaleDB_database(String timescaleDB_database) {
		this.timescaleDB_database = timescaleDB_database;
	}
	public String getTimescaleDB_password() {
		return timescaleDB_password;
	}
	public void setTimescaleDB_password(String timescaleDB_password) {
		this.timescaleDB_password = timescaleDB_password;
	}
	public String getTimescaleDB_username() {
		return timescaleDB_username;
	}
	public void setTimescaleDB_username(String timescaleDB_username) {
		this.timescaleDB_username = timescaleDB_username;
	}
	public String getTimescaleDB_url() {
		return timescaleDB_url;
	}
	public void setTimescaleDB_url(String timescaleDB_url) {
		this.timescaleDB_url = timescaleDB_url;
	}
	public String getTimescaleDB_port() {
		return timescaleDB_port;
	}
	public void setTimescaleDB_port(String timescaleDB_port) {
		this.timescaleDB_port = timescaleDB_port;
	}
	public String getTimescaleDB_connet_url() {
		String connect_url = "jdbc:postgresql://"+timescaleDB_url+":"+timescaleDB_port+"/"+timescaleDB_database;
		return connect_url;
	}
	
}

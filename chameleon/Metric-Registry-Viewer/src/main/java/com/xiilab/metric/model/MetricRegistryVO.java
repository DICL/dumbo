package com.xiilab.metric.model;

public class MetricRegistryVO {
	
	private Long num;
	private String name;
	private String col_name;
	private String description;
	private String pid_symbol;
	private String y_axis_label;
	private String parser_script;
	
	public Long getNum() {
		return num;
	}
	public void setNum(Long num) {
		this.num = num;
	}
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getDescription() {
		return description;
	}
	public void setDescription(String description) {
		this.description = description;
	}
	public String getPid_symbol() {
		return pid_symbol;
	}
	public void setPid_symbol(String pid_symbol) {
		this.pid_symbol = pid_symbol;
	}
	public String getY_axis_label() {
		return y_axis_label;
	}
	public void setY_axis_label(String y_axis_label) {
		this.y_axis_label = y_axis_label;
	}
	public String getParser_script() {
		return parser_script;
	}
	public void setParser_script(String parser_script) {
		this.parser_script = parser_script;
	}
	public String getCol_name() {
		return col_name;
	}
	public void setCol_name(String col_name) {
		this.col_name = col_name;
	}
}

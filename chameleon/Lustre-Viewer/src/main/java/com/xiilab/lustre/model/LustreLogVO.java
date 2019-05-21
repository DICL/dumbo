package com.xiilab.lustre.model;

public class LustreLogVO {
	 	private long num;
	 	private String log_type;
	    private String row_key;
	    private String data;
	    
	    private String create_date;
	    private String log_label;
	    
	    private String host_name;
	    
	    
		public long getNum() {
			return num;
		}
		public void setNum(long num) {
			this.num = num;
		}
		public String getLog_type() {
			return log_type;
		}
		public void setLog_type(String log_type) {
			this.log_type = log_type;
		}
		public String getRow_key() {
			return row_key;
		}
		public void setRow_key(String row_key) {
			this.row_key = row_key;
		}
		public String getData() {
			return data;
		}
		public void setData(String data) {
			this.data = data;
		}
		public String getCreate_date() {
			return create_date;
		}
		public void setCreate_date(String create_date) {
			this.create_date = create_date;
		}
		public String getLog_label() {
			return log_label;
		}
		public void setLog_label(String log_label) {
			this.log_label = log_label;
		}
		public String getHost_name() {
			return host_name;
		}
		public void setHost_name(String host_name) {
			this.host_name = host_name;
		}
}

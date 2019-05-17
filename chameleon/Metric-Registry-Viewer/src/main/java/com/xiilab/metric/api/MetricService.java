package com.xiilab.metric.api;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.TimeZone;
import java.util.UUID;

import javax.servlet.http.HttpServletRequest;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import com.github.jknack.handlebars.Handlebars;
import com.github.jknack.handlebars.Template;
import com.github.jknack.handlebars.io.ClassPathTemplateLoader;
import com.github.jknack.handlebars.io.TemplateLoader;
import com.xiilab.metric.model.AmbariYarnAppMonitorConfig;
import com.xiilab.metric.model.MetricRegistryVO;
import com.xiilab.metric.model.YarnAppNodesVO;
import com.xiilab.metric.utilities.AmbariProperty;
import com.xiilab.metric.utilities.SshClient;

/**
 * @author xiilab
 *
 */
@Service
public class MetricService {
	static final Logger LOGGER = LoggerFactory.getLogger(MetricService.class);
	
	// 제거할 특수문자
	public final static String MATCH = "[^\uAC00-\uD7A3xfe0-9a-zA-Z\\s]";
	@Autowired
	private MetricDAO metricDAO;
	@Autowired
	private SshClient sshClient;
	
	@Autowired
	private AmbariProperty ambariProperty;

	/**
	 * 메트릭 리스트 가져오기
	 * @return
	 */
	public List<MetricRegistryVO> getMetricList() {
		return metricDAO.getMetricList();
	}
	
	/**
	 * 테이블 체크
	 * @return
	 */
	public Boolean checkMetricRegistryTable() {
		return metricDAO.checkMetricRegistryTable().size() > 0;
	}
	
	/**
	 * create table and send script files
	 * @return
	 */
	public Boolean initTable() {
		
		try {
			// create crontab cycle
			metricDAO.create_metric_cycle_table();
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
		}
		try {
			// create metric table
			metricDAO.create_metric_registry_table();
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
		}
		try {
			// create sequence
			 metricDAO.create_timescale_sequence();
		}catch (Exception e) {
			LOGGER.error(e.getMessage());
		}
		
		try {
			// update crontab
			updateMetricCycleTime(3);
			create_hyper_table();
			return true;
		} catch (Exception e) {
			LOGGER.error(e.getMessage());
			return false;
		}
	}
	

	/**
	 * 메트릭 추가
	 * @param metric
	 * @return
	 */
	public Boolean addMetric(MetricRegistryVO metric) {
		// 특수문자 제거
		metric.setCol_name(metric.getName().replaceAll(MATCH, ""));
		return ( metricDAO.addMetric(metric) && create_hyper_table());
	}

	/**
	 * 메트릭 내용보기
	 * @param num
	 * @return
	 */
	public MetricRegistryVO View_Metric(Long num) {
		MetricRegistryVO metricRegistryVO = new MetricRegistryVO();
		metricRegistryVO.setNum(num);
		return metricDAO.viewMetric(metricRegistryVO);
	}

	/**
	 * 메트릭 업데이트
	 * @param metric
	 * @return
	 */
	public Boolean updateMetric(MetricRegistryVO metric) {
		// 특수문자 제거
		metric.setCol_name(metric.getName().replaceAll(MATCH, ""));
		return (metricDAO.updateMetric(metric)&& create_hyper_table() );
	}

	/**
	 * 메트릭 삭제
	 * @param metric
	 * @return
	 */
	public Boolean deleteMetric(MetricRegistryVO metric) {
		return (metricDAO.deleteMetric(metric) && create_hyper_table());
	}

	public String test() {
		return metricDAO.test_timescale();
	}
	
	/**
	 * 메트릭 수집용 파이썬 스크립트 제작
	 * @param hyper_table_name
	 * @param metric_list
	 * @param file_path
	 * @return
	 */
	public Boolean createPythonScriptFile(String hyper_table_name, List<MetricRegistryVO> metric_list, String file_path) {
		
//		HttpServletRequest request = ((ServletRequestAttributes)RequestContextHolder.currentRequestAttributes()).getRequest();
//		String root_path = request.getSession().getServletContext().getRealPath("/");
//		Writer writer = null;
		
		TemplateLoader loader = new ClassPathTemplateLoader();
		loader.setPrefix("/python_templates");
		loader.setSuffix(".py");
		Handlebars handlebars = new Handlebars(loader);
		
		
		AmbariYarnAppMonitorConfig configurations = ambariProperty.getAmbariYarnAppMonitorConfig();
		
		
		
//		Properties config = new Properties();
		
		// 프로퍼티 파일 읽어오기
//		InputStream inputStream = getClass().getClassLoader().getResourceAsStream("mybatis/jdbc.properties");
//		try {
//			config.load(inputStream);
//		} catch (IOException e) {
//			e.printStackTrace();
//			return false;
//		}
		Map<String,String> database = new HashMap<>();
//		String database_full_url = config.getProperty("jdbc.timescale.url");
//		String[] url_array = database_full_url.split("/");
//		String database_name = url_array[url_array.length - 1];
//		String database_url = url_array[url_array.length - 2].split(":")[0];
//		String database_port = url_array[url_array.length - 2].split(":")[1];
		
		
		String database_url = configurations.getTimescaleDB_url();
		String database_port = configurations.getTimescaleDB_port();
		String user_name = configurations.getTimescaleDB_username();
		String password = configurations.getTimescaleDB_password();
		String database_name = configurations.getTimescaleDB_database();
		
		database.put("url", database_url);
		database.put("port", database_port);
		database.put("user", user_name);
		database.put("password", password);
		database.put("neme", database_name);
		database.put("table_name", hyper_table_name);
		
		List<Map<String, String>> col_list = new ArrayList<>();
		for (MetricRegistryVO metric  : metric_list) {
			Map<String, String> temp_map = new HashMap<>();
			temp_map.put("col_name", metric.getCol_name());
			temp_map.put("parser", metric.getParser_script().replace("'", "\\'").replace(metric.getPid_symbol(), "'+PID+'") );
			col_list.add(temp_map);
		}
		
		
		Map<String,Object> table_info = new HashMap<>();
		table_info.put("col_list", col_list);
		table_info.put("database", database);
		
		
		try {
			Template template = handlebars.compile("yarn_appmonitor_send");
			String python_code = template.apply(table_info);
//			String python_file_path = root_path + file_path;
			String python_file_path = file_path;
//			String python_file_path = root_path + "/resources/python/yarn_appmonitor_send.py";
			LOGGER.info("result:\n{}\n",python_code);
			LOGGER.info("file path : {}",python_file_path);
			
			File file = new File(python_file_path);
			FileWriter fw = new FileWriter(file);
			fw.write(python_code);
			fw.close();
			
			return true;
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
		
	}
	
	
	/**
	 * 생성된 메트릭 리스트들을 읽어와서 새로운 하이퍼 테이블 생성
	 * @return
	 */
	public Boolean create_hyper_table() {
		if(! metricDAO.check_sequence()) {
			return false;
		}
		TimeZone tz;
		tz = TimeZone.getTimeZone("Asia/Seoul"); 
		DateFormat df = new SimpleDateFormat("yyyyMMddHHmmss");
		df.setTimeZone(tz);
		String time_label = df.format(new Date());
		String hyper_table_name = "apptable_" + metricDAO.get_sequence_next_val() +"_"+ time_label;
		
		
		if(metricDAO.find_hyper_table(hyper_table_name).size() <= 0) {
			List<MetricRegistryVO> metric_list = metricDAO.get_all_metric_list();
			Map<String,Object> table_info = new HashMap<>();
			table_info.put("colum_list", metric_list);
			table_info.put("table_name", hyper_table_name);
			
			Boolean result = metricDAO.create_timescale(table_info);
//			result = metricDAO.create_hypertable(table_info);
			
			HttpServletRequest request = ((ServletRequestAttributes)RequestContextHolder.currentRequestAttributes()).getRequest();
			String root_path = request.getSession().getServletContext().getRealPath("/");
			
			// 스크립트 파일 제작
			String client_file_path = "/tmp/yarn_appmonitor_send.py";
			result = createPythonScriptFile(hyper_table_name,metric_list,client_file_path);
			result = send_agent_file(client_file_path);
			return result;
		}else {
			return false;
		}
	}
	
	/**
	 * 파이선 코드를 클라이언트에 전송
	 * @param client_file_path
	 * @return
	 */
	private Boolean send_agent_file(String client_file_path) {
		List<YarnAppNodesVO> node_list = metricDAO.getYarnAppClientNodes();
		String remoteDir = metricDAO.getYarnAppMonitorClientFilePath();
		try {
			File agent_file = new File(client_file_path);
			for (YarnAppNodesVO yarn_node : node_list) {
				sshClient.sftpSend(yarn_node, remoteDir, agent_file);
				sshClient.execjsch(yarn_node, "service crond restart");
			}
		} catch (Exception e) {
			LOGGER.error(e.getMessage());
			return false;
		}
		
		return true;
	}

	public List<YarnAppNodesVO> getYarnAppClientNodes() {
		return metricDAO.getYarnAppClientNodes();
	}

	/**
	 * 현재 크론탭 주기 가져오기
	 * @return
	 */
	public Map<String, Object> getMetricCycleTime() {
		return metricDAO.getMetricCycleTime();
	}

	/**
	 * 크론탭 주기 업데이트
	 * @param cycle_time
	 * @return
	 */
	public Map<String, Object> updateMetricCycleTime(Integer cycle_time) {
		Map<String, Object> result = new HashMap<>();
		
		String remoteDir = metricDAO.getYarnAppMonitorClientFilePath();
		
		boolean update_result = false;
		String crontab_file_path = "/tmp/crontab";
		String python_script_path = remoteDir+"yarn_appmonitor_send.py";
		if(createCronTabScriptFile(cycle_time,crontab_file_path,python_script_path) && send_crontab_file(crontab_file_path)) {
			update_result = metricDAO.updateMetricCycleTime(cycle_time) > 0;
		}else {
			update_result = false;
		}
		result.put("status", update_result);
		return result;
	}
	
	
	/**
	 * 각 클라이언트로 파일전송후 크론탭 재시작
	 * @param crontab_file_path
	 * @return
	 */
	public Boolean send_crontab_file(String crontab_file_path) {
		List<YarnAppNodesVO> node_list = metricDAO.getYarnAppClientNodes();
		String remoteDir = "/etc/";
		try {
			File crontab_file = new File(crontab_file_path);
			for (YarnAppNodesVO yarn_node : node_list) {
				sshClient.sftpSend(yarn_node, remoteDir, crontab_file);
				sshClient.execjsch(yarn_node, "service crond restart");
			}
		} catch (Exception e) {
			LOGGER.error(e.getMessage());
			return false;
		}
		return true;
	}

	/**
	 * 크론탭 파일 생성
	 * @param cycle_time
	 * @param crontab_file_path
	 * @param python_script_path
	 * @return
	 */
	public Boolean createCronTabScriptFile(Integer cycle_time,String crontab_file_path,String python_script_path) {
		int one_minute_second = 60;
		
		Map<String,Object> handlebars_set_data = new HashMap<>();
		List<Map<String,String>> handlebars_command_list = new ArrayList<>();
		
		TemplateLoader loader = new ClassPathTemplateLoader();
		loader.setPrefix("/crontab_templates");
		loader.setSuffix(".hbs");
		Handlebars handlebars = new Handlebars(loader);
		
		for (int set_time = cycle_time; set_time < one_minute_second; set_time += cycle_time) {
			Map<String,String> temp_set_data = new HashMap<>();
			temp_set_data.put("time", set_time+"");
			temp_set_data.put("file_path", python_script_path);
			handlebars_command_list.add(temp_set_data);
		}
		handlebars_set_data.put("command_list", handlebars_command_list);
		handlebars_set_data.put("file_path", python_script_path);
		
		try {
			Template template = handlebars.compile("crontab");
			String crontab_code = template.apply(handlebars_set_data);
			LOGGER.info("result:\n{}\n",crontab_code);
			LOGGER.info("file path : {}",crontab_file_path);
			
			File file = new File(crontab_file_path);
			FileWriter fw = new FileWriter(file);
			fw.write(crontab_code);
			fw.close();
			
		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}	
		return true;
	}
}

package org.kisti.moha;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class MOHA_KafkaStop {
	private static MOHA_Zookeeper zk;

	public static void main(String[] args) {
		MOHA_KafkaStop stop = new MOHA_KafkaStop();
		stop.run();
	}

	public MOHA_KafkaStop() {
		Properties prop = new Properties();
		/* Loading MOHA.Conf File */
		String kafka_libs = "/usr/hdp/kafka_2.11-0.9.0.0/libs/*";
		String kafkaVersion = "kafka_2.11-0.10.1.0";
		String kafkaClusterId = "KafkaCluster";
		String zookeeperServer = "localhost";
		try {
			prop.load(new FileInputStream("conf/MOHA.conf"));
			kafka_libs = prop.getProperty("MOHA.dependencies.kafka.libs");
			kafkaVersion = prop.getProperty("MOHA.dependencies.kafka.version");
			kafkaClusterId = prop.getProperty("MOHA.kafka.cluster.id");
			zookeeperServer = prop.getProperty("MOHA.zookeeper.server.address");
			System.out.println(kafka_libs);
			System.out.println(kafkaVersion);
			System.out.println(kafkaClusterId);
			System.out.println(zookeeperServer);
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Map<String, String> newenv = new HashMap<>();
		newenv.put(MOHA_Properties.CONF_ZOOKEEPER_SERVER,zookeeperServer);
		try {
			setEnv(newenv);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		 
		zk = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT , kafkaClusterId);
	}


	public void run() {

		zk.setStopRequest(true);

	}
	
	protected static void setEnv(Map<String, String> newenv) throws Exception {
		  try {
		    Class<?> processEnvironmentClass = Class.forName("java.lang.ProcessEnvironment");
		    Field theEnvironmentField = processEnvironmentClass.getDeclaredField("theEnvironment");
		    theEnvironmentField.setAccessible(true);
		    Map<String, String> env = (Map<String, String>) theEnvironmentField.get(null);
		    env.putAll(newenv);
		    Field theCaseInsensitiveEnvironmentField = processEnvironmentClass.getDeclaredField("theCaseInsensitiveEnvironment");
		    theCaseInsensitiveEnvironmentField.setAccessible(true);
		    Map<String, String> cienv = (Map<String, String>)     theCaseInsensitiveEnvironmentField.get(null);
		    cienv.putAll(newenv);
		  } catch (NoSuchFieldException e) {
		    Class[] classes = Collections.class.getDeclaredClasses();
		    Map<String, String> env = System.getenv();
		    for(Class cl : classes) {
		      if("java.util.Collections$UnmodifiableMap".equals(cl.getName())) {
		        Field field = cl.getDeclaredField("m");
		        field.setAccessible(true);
		        Object obj = field.get(env);
		        Map<String, String> map = (Map<String, String>) obj;
		        map.clear();
		        map.putAll(newenv);
		      }
		    }
		  }
		}
}

package org.kisti.moha;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Properties;

import org.apache.hadoop.fs.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class ConfigureFileTest {
	// private static final Logger LOG =
	// LoggerFactory.getLogger(ConfigureFileTest.class);

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// Properties prop = new Properties();
		// /* Loading MOHA.Conf File */
		// try {
		// prop.load(new FileInputStream("conf/MOHA.conf"));
		// String kafka_libs = prop.getProperty("MOHA.dependencies.kafka.libs");
		// System.out.println(kafka_libs);
		// } catch (FileNotFoundException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// } catch (IOException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }

		long startingTime = System.currentTimeMillis();
		long runningTime = 0;

		for (int i = 0; i < 10; i++) {
			startingTime = System.currentTimeMillis();
			run();
			runningTime = System.currentTimeMillis() - startingTime;
			// LOG.info("Time = {} is {}",i,String.valueOf(runningTime));
			System.out.println(runningTime);
		}

		/*
		 * Reading in the URL of the DBManager in the configuration file e.g.,
		 * DBManager.Address=http://150.183.158.172:9000/Database
		 */

	}

	private static void run() {
		long i = 0;
		while (i < 1000000) {
			long j = 0;
			while (j < 10000) {
				j++;
			}
			i++;
		}
	}

}

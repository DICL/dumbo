package org.kisti.moha;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.common.PartitionInfo;
import org.apache.kafka.common.TopicPartition;

public class MOHA_TaskExecutor {

	private static final int SESSION_MAXTIME = 12* 60 * 60 * 1000; // This is about 12 hours 
	private static final int EXTENDED_SESSION_TIME = 10 * 1000;// Ten seconds

	private YarnConfiguration conf;

	private static MOHA_Queue jobQueue;

	private static MOHA_Logger LOG;

	private MOHA_ExecutorInfo info;
	private MOHA_Database database;

	public MOHA_TaskExecutor(String[] args) throws IOException {
		// TODO Auto-generated constructor stub
		/* Save input parameters */
		info = new MOHA_ExecutorInfo();
		info.setAppId(args[0]);
		info.setContainerId(args[1]);
		info.setExecutorId(Integer.parseInt(args[2]));
		info.setHostname(NetUtils.getHostname());
		info.setLaunchedTime(System.currentTimeMillis());
		info.setQueueName(info.getAppId());
		
		String zookeeperConnect = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_CONNECT);
		String bootstrapServer = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER);
		
		LOG = new MOHA_Logger(MOHA_TaskExecutor.class, Boolean.parseBoolean(info.getConf().getKafkaDebugEnable()), info.getConf().getDebugQueueName(),
				zookeeperConnect, bootstrapServer , info.getAppId());
		LOG.register();

		LOG.all("TaskExecutor [" + info.getExecutorId() + "] is started in " + info.getHostname());

		LOG.debug(info.toString());

		conf = new YarnConfiguration();
		FileSystem.get(conf);
		LOG.debug(conf.toString());

		//String zookeeperConnect = System.getenv(MOHA_Properties.CONF_ZOOKEEPER_CONNECT);
		//String bootstrapServer = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_KAFKA, info.getConf().getKafkaClusterId()).getBootstrapServers();
		


		jobQueue = new MOHA_Queue(zookeeperConnect, bootstrapServer, info.getQueueName());
		database = new MOHA_Database(Boolean.parseBoolean(info.getConf().getMysqlLogEnable()));
		LOG.debug(database.toString());
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// LOG.info("Container just started on {}" + NetUtils.getHostname());

		try {
			MOHA_TaskExecutor executor = new MOHA_TaskExecutor(args);
			executor.run();
			// executor.KOHA_run();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	private void KOHA_producer(int num, String msg) {
		/* Creating a queue and pushing jobs to the queue */
		jobQueue.register();
		// add padding
		// LOG.all("TaskExecutor [" + info.getExecutorId() + "] adds padding");

		MOHA_Zookeeper zks = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_MOHA, info.getAppId());
		// ready for pushing
		// for (int i = 0; i < num*1; i++) {
		// jobQueue.push(Integer.toString(i), msg);
		// }
		zks.setReadyfor(info.getExecutorId(), MOHA_Properties.TIMMING_PUSHING);

		// LOG.all("TaskExecutor [" + info.getExecutorId() + "] is ready");
		while (zks.getTimming() != MOHA_Properties.TIMMING_PUSHING) {
			// wait for start
			// for (int i = 0; i < num; i++) {
			// jobQueue.push(Integer.toString(i), msg);
			// }
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		// calculate
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] starts actual pushing");
		long startTime = System.currentTimeMillis();

		for (int i = 0; i < num; i++) {
			jobQueue.push(Integer.toString(i), msg);
		}
		long pushingTime = (System.currentTimeMillis() - startTime);
		long rate = (num / pushingTime) * 1000;
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] have pushed " + String.valueOf(num) + " (messages) in " + String.valueOf(pushingTime)
				+ " mini seconds and speed is " + String.valueOf(rate) + " messages/second");
		info.setPushingRate(rate);// messages per second

		// add padding
		// LOG.all("TaskExecutor [" + info.getExecutorId() + "] keeps padding");

		zks.setReadyfor(info.getExecutorId(), MOHA_Properties.TIMMING_PUSHING_FINISH);

		while (zks.getTimming() != MOHA_Properties.TIMMING_PUSHING_FINISH) {
			try {
				Thread.sleep(10);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			// for (int i = 0; i < num; i++) {
			// jobQueue.push(Integer.toString(i), msg);
			// }
		}

		LOG.all("TaskExecutor [" + info.getExecutorId() + "] producer stops");
	}

	private void KOHA_consumer(int number) {
		long startTime = System.currentTimeMillis();
		long expiredTime = System.currentTimeMillis() + 10 * EXTENDED_SESSION_TIME;

		long pollingTime = 0;
		Boolean found = false;
		int num = 0;
		ConsumerRecords<String, String> records;
		MOHA_Zookeeper zks = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_MOHA, info.getAppId());
		num = 0;
		// LOG.all("TaskExecutor [" + info.getExecutorId() + "] adds padding for
		// fetching");
		jobQueue.subcribe();
		while (zks.getTimming() != MOHA_Properties.TIMMING_FETCHING) {
			for (int i = 0; i < 10; i++) {
				records = jobQueue.poll(100);

				num += records.count();
			}
			if (!found && (num > number)) {
				LOG.all("TaskExecutor [" + info.getExecutorId() + "] got first message");
				zks.setReadyfor(info.getExecutorId(), MOHA_Properties.TIMMING_FETCHING);
				found = true;
			}

		}
		// start
		// ......................................................................................................

		num = 0;
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] starts for actual fetching");
		startTime = System.currentTimeMillis();
		while (num < number) {

			records = jobQueue.poll(100);

			if (records.count() > 0) {

				num += records.count();

				// LOG.debug("TaskExecutor [" + info.getExecutorId() + "] got "
				// + String.valueOf(records.count()));
			}

			// for (ConsumerRecord<String, String> record : records) {
			// LOG.debug("TaskExecutor [" + info.getExecutorId() + "]: " +
			// record.toString());
			// }
		}
		pollingTime = System.currentTimeMillis() - startTime;

		if (pollingTime > 0) {
			long rate = (num / pollingTime) * 1000;
			LOG.all("TaskExecutor [" + info.getExecutorId() + "] have fetched " + String.valueOf(num) + " (messages) in " + String.valueOf(pollingTime)
					+ " mini seconds and speed is " + String.valueOf(rate) + " messages/second");
			info.setPollingRate(rate);
		} else {
			LOG.all("TaskExecutor [" + info.getExecutorId() + "] did not get any messages ");
			info.setPollingRate(0);
		}
		info.setNumExecutedTasks(num);

		// FINISH MEASURING

		zks.setReadyfor(info.getExecutorId(), MOHA_Properties.TIMMING_FETCHING_FINISH);

		///////////////////////// padding

		while (zks.getTimming() != MOHA_Properties.TIMMING_FETCHING_FINISH) {
			// LOG.debug("TaskExecutor [" + info.getExecutorId() + "] before
			// polling ");
			for (int i = 0; i < 100; i++) {
				records = jobQueue.poll(100);
				num += records.count();
			}

		}
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] consumer stops");
	}

	private void KOHA_run() {
		// String msg = "qwertyuioplkjhgfdsazxcvbnm";
		String msg_unit = "sleep 0";
		String msg = "";
		for (int i = 0; i < 1; i++) {
			msg += msg_unit;
		}

		int num = 500000;

		LOG.all("TaskExecutor [" + info.getExecutorId() + "] starts calling producer: " + msg);
		LOG.all("Length = " + String.valueOf(msg.length()));

		KOHA_producer(num, msg);

		LOG.all("TaskExecutor [" + info.getExecutorId() + "] starts calling consumer");
		// KOHA_consumer(num);
		consumer1();

		// database.insertExecutorInfoToDatabase(info);
		jobQueue.close();
		jobQueue.deleteQueue();
	}

	private void consumer1() {
		// TODO Auto-generated method stub

		long startingTime = System.currentTimeMillis();
		long expiredTime = System.currentTimeMillis() + 10 * EXTENDED_SESSION_TIME;
		int numOfCommands = 0;
		int numOfPolls = 0;
		int retries = 20;
		boolean found = false; // get first message from the queue
		info.setEndingTime(startingTime);
		LOG.all(info.toString());
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] starts polling and processing jobs got from the job queue");

		MOHA_Zookeeper zkServer = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_MOHA, info.getAppId());
		// MOHA_Zookeeper zks = new
		// MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_MOHA, info.getAppId());

		LOG.debug(zkServer.toString());
		jobQueue.subcribe();

		while (System.currentTimeMillis() < expiredTime) {

			ConsumerRecords<String, String> records = jobQueue.poll(100);

			if ((records.count() > 0) && (!found)) {
				info.setFirstMessageTime(System.currentTimeMillis());
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] got first messages at tries " + (20 - retries));
				retries = 5;
				found = true;
			}

			if ((records.count() > 0) && ((expiredTime - startingTime) < SESSION_MAXTIME)) {

				jobQueue.commitSync();
				numOfPolls++;
				numOfCommands += records.count();
				info.setEndingTime(System.currentTimeMillis());
				// expiredTime = System.currentTimeMillis() +
				// EXTENDED_SESSION_TIME;

				zkServer.setCompletedJobsNum(info.getExecutorId(), numOfCommands);

				// LOG.all("TaskExecutor [" + info.getExecutorId() + "] have
				// completed " + numOfCommands + " jobs");
			} else if (retries > 0) {

				// jobQueue.subcribe();
				// LOG.all("TaskExecutor [" + info.getExecutorId() + "] could no
				// longer get meseages, try to re-poll");
				try {
					Thread.sleep(100);
					// jobQueue.commitSync();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				retries--;
				// expiredTime = System.currentTimeMillis() +
				// EXTENDED_SESSION_TIME;
			}

			if (!found) {
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] could not found");
			}

			if (retries == 0) {
				zkServer.setReadyfor(info.getExecutorId(), MOHA_Properties.TIMMING_FETCHING_FINISH);
			}

			///////////////////////// padding

			if (zkServer.getTimming() != MOHA_Properties.TIMMING_FETCHING_FINISH) {
				expiredTime = System.currentTimeMillis() + EXTENDED_SESSION_TIME;
			}

		}
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] have completed " + numOfCommands + " jobs");

		try {
			Thread.sleep(2000);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		long executingTime = info.getEndingTime() - info.getFirstMessageTime();
		info.setExecutionTime(executingTime);
		info.setNumExecutedTasks(numOfCommands);
		info.setNumOfPolls(numOfPolls);
		info.setEndingTime(System.currentTimeMillis());
		LOG.debug(info.toString());
		database.insertExecutorInfoToDatabase(info);
		LOG.info(database.toString());
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] exists");
		jobQueue.close();
	}

	public void copyFromHdfs(String source, String dest) throws IOException {

		Configuration conf = new Configuration();

		FileSystem fileSystem = FileSystem.get(conf);
		Path srcPath = new Path(source);

		Path dstPath = new Path(dest);
		LOG.debug("source " + srcPath.toString());
		LOG.debug("dstPath " + dstPath.toString());
		// Check if the file already exists
		if (!(fileSystem.exists(dstPath))) {
			LOG.debug("No such destination " + dstPath);
			return;
		}

		// Get the filename out of the file path
		String filename = source.substring(source.lastIndexOf('/') + 1, source.length());

		try {
			fileSystem.copyToLocalFile(srcPath, dstPath);
			LOG.debug("File " + filename + "copied to " + dest);
		} catch (Exception e) {
			LOG.debug("Exception caught! :" + e);
			System.exit(1);
		} finally {
			fileSystem.close();
		}
	}

	private void run() {
		// TODO Auto-generated method stub

		long startingTime = System.currentTimeMillis();
		long expiredTime = System.currentTimeMillis() + 6 * EXTENDED_SESSION_TIME;
		int numOfCommands = 0;
		int numOfPolls = 0;
		int retries = 2;
		boolean found = false; // get first message from the queue
		/*Set ending time, which can be updated frequently*/
		info.setEndingTime(startingTime);
				
		LOG.debug("TaskExecutor [" + info.getExecutorId() + "] starts polling and processing jobs got from the job queue");
		/*Construct zookeeper server which is used to inform number of completed tasks to MOHA Client */
		MOHA_Zookeeper zkServer = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_MOHA, info.getAppId());

		/*Subscribe to the job queue to be allowed polling task input files*/
		jobQueue.subcribe();
		
		Configuration conf = new Configuration();
		FileSystem fileSystem = null;

		try {
			fileSystem = FileSystem.get(conf);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		Path srcPath;
		Path dstPath;

		while (System.currentTimeMillis() < expiredTime) {
			/*Poll task input file location in HDFS*/
			ConsumerRecords<String, String> records = jobQueue.poll(100);			

			if ((records.count() > 0) && (!found)) {
				info.setFirstMessageTime(System.currentTimeMillis());
				LOG.all("TaskExecutor [" + info.getExecutorId() + "] found first package" + "& partitions: " + jobQueue.assignment().toString());
				found = true;
			}

			for (ConsumerRecord<String, String> record : records) {
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "]: " + record.toString() + jobQueue.assignment().toString());
				/* Extract the location of input file in HDFS */
				srcPath = new Path(record.value());
				dstPath = new Path(".");
				// LOG.debug("source " + srcPath.toString());
				// LOG.debug("dstPath " + dstPath.toString());

				try {
					/* Copy tar file from HDFS to local directory */
					fileSystem.copyToLocalFile(srcPath, dstPath);
					LOG.debug("source " + srcPath.getName());
					/* Uncompress tar file to tmp folder */
					List<String> uncompressCommand = new ArrayList<String>();
					uncompressCommand.add("tar");
					uncompressCommand.add("-xvzf");
					uncompressCommand.add(srcPath.getName());
					uncompressCommand.add("-C");
					uncompressCommand.add("tmp");
					LOG.debug(uncompressCommand.toString());
					ProcessBuilder builder = new ProcessBuilder(uncompressCommand);
					Process p;
					String cliResponse;

					p = builder.start();
					p.waitFor();
					BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
					while ((cliResponse = br.readLine()) != null) {
						// LOG.debug(line);
					}
					/*
					 * Copy all files from executable directory to current
					 * directory
					 */
					File[] listFiles = new File(MOHA_Properties.EXECUTABLE_DIR).listFiles();
					for (File file : listFiles) {
						if (file.isFile()) {
							file.renameTo(new File(file.getName()));
							file.delete();
						}
					}
					/* Get the list of folders, which is the input of tasks */
					File[] listFolders = new File("tmp").listFiles();

					for (File folder : listFolders) {
						Long begin = System.currentTimeMillis();
						if (folder.isDirectory()) {
							LOG.debug("Input directory for current task:" + folder.getName());
							/* Build command to execute the task */
							List<String> taskCommand = new ArrayList<String>();
							taskCommand.add("./autodock_vina.sh");
							taskCommand.add("5-FU");
							taskCommand.add("tmp/" + folder.getName());
							taskCommand.add("scPDB_coordinates.tsv");
							LOG.debug("Command for current task:" + taskCommand.toString());
							builder = new ProcessBuilder(taskCommand);
							/* Execute the task */
							p = builder.start();
							p.waitFor();
							BufferedReader buffReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
							while ((cliResponse = buffReader.readLine()) != null) {
								//LOG.debug(cliResponse);
							}
							LOG.all("TaskExecutor [" + info.getExecutorId() + "]["+record.partition()+"] Execution time [" + folder.getName() + "]: "
									+ String.valueOf(System.currentTimeMillis() - begin));
							/*
							 * Delete input folder after the task is completed
							 */
							FileUtil.fullyDelete(folder);
						}
					}
					/* Delete the tar tgz file after uncompressed */
					//fileSystem.delete(new Path(srcPath.getName()), true);
					FileUtil.fullyDelete(new File(srcPath.getName()));
				} catch (IOException | InterruptedException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}

				// List<String> command = new ArrayList<String>();
				//
				// String[] str = record.value().split(" ");
				// for (String cmd : str) {
				// command.add(cmd);
				// }
				// //LOG.all("TaskExecutor [" + info.getExecutorId() + "]: " +
				// command.toString());
				// ProcessBuilder builder = new ProcessBuilder(command);
				// Process p;
				// String line;
				// try {
				// p = builder.start();
				// p.waitFor();
				// BufferedReader br = new BufferedReader(new
				// InputStreamReader(p.getInputStream()));
				// while ((line = br.readLine()) != null) {
				// LOG.debug("Task Executor (" + info.getExecutorId() + ") " +
				// "from partition "
				// + record.partition() + " : " + line);
				// }
				// } catch (IOException e) {
				// // TODO Auto-generated catch block
				// e.printStackTrace();
				// } catch (InterruptedException e) {
				// // TODO Auto-generated catch block
				// e.printStackTrace();
				// }
			}

			if ((records.count() > 0) && ((expiredTime - startingTime) < SESSION_MAXTIME)) {

				jobQueue.commitSync();
				numOfPolls++;
				numOfCommands += records.count();
				info.setEndingTime(System.currentTimeMillis());
				expiredTime = System.currentTimeMillis() + EXTENDED_SESSION_TIME;
				zkServer.setCompletedJobsNum(info.getExecutorId(), numOfCommands);
				// LOG.debug("TaskExecutor [" + info.getExecutorId() + "] have
				// completed " + numOfCommands + " jobs");
			} else if ((retries > 0) && found) {
				/*This case happens when it's no longer to poll messages, need to try for a couple of time before actually quit*/
				try {
					Thread.sleep(1000);
					jobQueue.commitSync();
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				retries--;
				expiredTime = System.currentTimeMillis() + EXTENDED_SESSION_TIME;
			}

		}
		/*Display list of files and directories for debugging*/
		File curDir = new File(".");
		getAllFiles(curDir);
		
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] have completed " + numOfCommands + " tasks");

		long executingTime = info.getEndingTime() - info.getFirstMessageTime();
		
		info.setExecutionTime(executingTime);
		info.setNumExecutedTasks(numOfCommands);
		info.setNumOfPolls(numOfPolls);
		info.setEndingTime(System.currentTimeMillis());
		LOG.debug(info.toString());
		database.insertExecutorInfoToDatabase(info);
		LOG.info(database.toString());
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] exists");
		jobQueue.close();
	}

	private static void getAllFiles(File curDir) {

		File[] filesList = curDir.listFiles();

		for (File f : filesList) {

			if (f.isDirectory()) {
				LOG.debug("Directory " + f.getName());
				getAllFiles(f);
			}
			if (f.isFile()) {
				LOG.debug(curDir.getPath() + "/" + f.getName());
			}
		}

	}
}

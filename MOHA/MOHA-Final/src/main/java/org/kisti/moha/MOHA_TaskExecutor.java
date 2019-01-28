package org.kisti.moha;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import javax.jms.JMSException;
import javax.jms.Message;
import javax.jms.TextMessage;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.LocatedFileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.net.NetUtils;
import org.apache.hadoop.util.SequentialNumber;
import org.apache.hadoop.yarn.conf.YarnConfiguration;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.kafka.common.PartitionInfo;
import org.apache.kafka.common.TopicPartition;

public class MOHA_TaskExecutor {

	private YarnConfiguration conf;
	// private static MOHA_Queue jobQueue;
	private static MOHA_Logger LOG;
	private volatile MOHA_ExecutorInfo info;
	private MOHA_Database database;
	public MOHA_TaskExecutor(String[] args) throws IOException {
		// TODO Auto-generated constructor stub
		/* Save input parameters */
		info = new MOHA_ExecutorInfo();
		info.setAppId(args[0]);
		info.setContainerId(args[1]);
		info.setExecutorId(Integer.parseInt(args[2]));
		info.setHostname(NetUtils.getHostname());
		info.setLaunchedTime(0);
		info.setQueueName(info.getAppId());

		String zookeeperConnect = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_CONNECT);
		String bootstrapServer = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER);

		LOG = new MOHA_Logger(MOHA_TaskExecutor.class, Boolean.parseBoolean(info.getConf().getKafkaDebugEnable()), info.getConf().getDebugQueueName(), zookeeperConnect, bootstrapServer,
				info.getAppId(), info.getExecutorId());
		LOG.register();

		LOG.all("TaskExecutor [" + info.getExecutorId() + "] is started in " + info.getHostname());

		LOG.debug(info.toString());

		conf = new YarnConfiguration();
		FileSystem.get(conf);
		LOG.debug(conf.toString());

		// String zookeeperConnect =
		// System.getenv(MOHA_Properties.CONF_ZOOKEEPER_CONNECT);
		// String bootstrapServer = new
		// MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT_KAFKA,
		// info.getConf().getKafkaClusterId()).getBootstrapServers();

		// jobQueue = new MOHA_Queue(zookeeperConnect, bootstrapServer,
		// info.getQueueName());

		database = new MOHA_Database(Boolean.parseBoolean(info.getConf().getMysqlLogEnable()));
		LOG.debug(database.toString());
	}

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		System.out.println("Container just started on {}" + NetUtils.getHostname());
		try {
			MOHA_TaskExecutor executor = new MOHA_TaskExecutor(args);
			// executor.run();
			executor.main_thread();
			// executor.KOHA_run();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void downloadInputDataFromHDFS(String source, String dest) throws IOException {

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
	/*This function parsers the task description, constructs task command and executes it*/
	private boolean executeTask(String task_description) {

		Configuration conf = new Configuration();
		FileSystem fs = null;
		ProcessBuilder builder;
		Process p;
		String cliResponse;
		boolean isSuccess = true;

		long begin;

		try {
			fs = FileSystem.get(conf);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		String hdfsHomeDirectory = System.getenv(MOHA_Properties.HDFS_HOME_DIRECTORY);
		// LOG.all("TaskExecutor [" + info.getExecutorId() + "] records:" + record);
		begin = System.currentTimeMillis();
		String[] command_compo = task_description.split(" ");
		for (int i = 0; i < command_compo.length - 1; i++) {
			// LOG.all(command_compo[i]);
		}
		List<String> taskCommand = new ArrayList<String>();
		String run_command = "./";
		run_command += command_compo[2 * Integer.valueOf(command_compo[0]) + 1];
		taskCommand.add(run_command);

		for (int i = 1; i <= Integer.valueOf(command_compo[0]); i++) {
			taskCommand.add(command_compo[i * 2]);
			if ((command_compo[i * 2 - 1].equals("D")) && (!new File(command_compo[i * 2]).exists())) {
				Path srcPath = new Path(hdfsHomeDirectory, "MOHA/" + info.getAppId() + "/" + command_compo[i * 2]);
				Path dstPath = new Path(command_compo[i * 2]);

				try {
					fs.copyToLocalFile(srcPath, dstPath);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					LOG.all("TaskExecutor [" + info.getExecutorId() + "] copyToLocalFile error: " + e.toString());
				}
			}
		}

		// LOG.debug("Command for current task:" + taskCommand.toString());
		builder = new ProcessBuilder(taskCommand);
		/* Execute the task */
		try {
			p = builder.start();

			p.waitFor();

			BufferedReader buffReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
			while ((cliResponse = buffReader.readLine()) != null) {
				 //LOG.debug("TaskExecutor [" + info.getExecutorId() + "] " + cliResponse);
			}
		} catch (IOException | InterruptedException e) {
			// TODO Auto-generated catch block
			LOG.all("TaskExecutor [" + info.getExecutorId() + "] copyToLocalFile error: " + e.toString());
			/*
			 * Copy all files from executable directory to current directory
			 */

			File[] listFiles = new File(MOHA_Properties.EXECUTABLE_DIR).listFiles();
			for (File file : listFiles) {
				if (file.isFile()) {
					file.renameTo(new File(file.getName()));
					// file.delete();
					LOG.debug("TaskExecutor [" + info.getExecutorId() + "] copy file: " + file.getName());
				}
			}

			e.printStackTrace();
			isSuccess = false;
		}

		
		// LOG.all("TaskExecutor [" + info.getExecutorId() + "][" + record + "] Execution time " + tempDockingTime);
		return isSuccess;
	}

	private void main_thread() {
		// TODO Auto-generated method stub

		long systemCurrentTime;
		int numOfCommands = 0;
		int numOfCommands_pre = 0;
		long capturing_time;
		int numOfPolls = 0;

		boolean found = false; // get first message from the queue
		String logs = "";

		LOG.debug("TaskExecutor [" + info.getExecutorId() + "] starts polling and processing jobs got from the job queue");
		/*
		 * Construct a Zookeeper server which is used to inform number of completed tasks to MOHA Client
		 */
		MOHA_Zookeeper zkServer = new MOHA_Zookeeper(MOHA_Properties.ZOOKEEPER_ROOT, info.getAppId());

		systemCurrentTime = zkServer.getSystemTime();
		capturing_time = systemCurrentTime;
		/* Set ending time, which can be updated frequently */
		info.setEndingTime(systemCurrentTime);
		info.setFirstMessageTime(systemCurrentTime);
		info.setLaunchedTime(systemCurrentTime);

		zkServer.createExecutorResultsDir(info.getExecutorId());
		String appType = System.getenv(MOHA_Properties.APP_TYPE);

		/*
		 * Copy all files from executable directory to current directory
		 */
		if (appType.equals("S")) {
			File[] listFiles = new File(MOHA_Properties.EXECUTABLE_DIR).listFiles();
			LOG.debug("TaskExecutor [" + info.getExecutorId() + "] number of files: " + listFiles.length);
			// Hadoop resource localization fail
			if (listFiles.length == 0) {
				LOG.all("TaskExecutor [" + info.getExecutorId() + "] is running on " + info.getHostname() + " resource localization fail ");
				String localResource = zkServer.getLocalResource();
				// LOG.debug("TaskExecutor [" + info.getExecutorId() + "] localResource: " + localResource);
				// String hdfsHomeDirectory = System.getenv(MOHA_Properties.HDFS_HOME_DIRECTORY);
				Path srcPath = new Path(localResource);
				Path dstPath = new Path("run.tgz");
				Configuration conf = new Configuration();
				FileSystem fs = null;

				try {
					fs = FileSystem.get(conf);
					fs.copyToLocalFile(srcPath, dstPath);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
					LOG.all(" copyToLocalFile error: " + e.toString());
				}
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Start uncompressing the tar file");
				getListOfFiles(info.getExecutorId(), new File("."));
				/* Uncompress tar file to tmp folder */
				List<String> uncompressCommand = new ArrayList<String>();
				uncompressCommand.add("tar");
				uncompressCommand.add("-xvzf");
				uncompressCommand.add("run.tgz");
				// uncompressCommand.add("-C");
				// uncompressCommand.add(MOHA_Properties.EXECUTABLE_DIR);
				LOG.debug(uncompressCommand.toString());
				ProcessBuilder builder = new ProcessBuilder(uncompressCommand);
				Process p;
				String cliResponse;

				try {
					p = builder.start();
					p.waitFor();
					BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
					while ((cliResponse = br.readLine()) != null) {
						LOG.debug("TaskExecutor [" + info.getExecutorId() + "] local resource files: " + cliResponse);
					}
				} catch (IOException | InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] after uncompress: ");
				getListOfFiles(info.getExecutorId(), new File("."));
			} else {
				LOG.all("TaskExecutor [" + info.getExecutorId() + "] is running on " + info.getHostname() + " resource localization success");
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Start copying files from exe to current directory");

				listFiles = new File(MOHA_Properties.EXECUTABLE_DIR).listFiles();
				for (File file : listFiles) {
					if (file.isFile()) {
						file.renameTo(new File(file.getName()));
						// file.delete();
						LOG.debug("TaskExecutor [" + info.getExecutorId() + "] copy file: " + file.getName());
					}
				}
			}
		}

		// Wait until all executors started
		while (!zkServer.getPollingEnable()) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				LOG.all(e.toString());
				e.printStackTrace();
			}

		}
		LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Start executor threads");
		
		/* Start executor threads */
		info.setNumRequestedThreads(60);
		info.setNumSpecifiedTasks(2);
		info.setNumExecutedTasks(0);
		info.setNumExecutingTasks(0);
		info.setNumActiveThreads(0);
		info.setQueueEmpty(false);

		LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Number of executing tasks: " + info.getNumExecutingTasks());
		for (int i = 0; i < info.getNumRequestedThreads(); i++) {
			ThreadTaskExecutor taskExecutor = new ThreadTaskExecutor(i);
			Thread startExecutor = new Thread(taskExecutor);
			startExecutor.start();
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		int numPreviousCompletedTasks;
		int numCompletedTasksPerWindow;
		long previousCheckingPoint, windowsTime;
		float previousPerformance, currentPerformance;
		int previousNumExecutingTasks = info.getNumExecutingTasks();
		
		
		/* Wait for at least one thread started */
		while (info.getNumRunningThreads() == 0) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if(previousNumExecutingTasks!= info.getNumExecutingTasks()){
				previousNumExecutingTasks = info.getNumExecutingTasks();
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Number of executing tasks: " + info.getNumExecutingTasks());
			}
		}
		
		/* Wait for fist windows */
		previousCheckingPoint = System.currentTimeMillis();
		while (info.getNumExecutedTasks() < 2 * info.getNumSpecifiedTasks()) {
			try {
				Thread.sleep(100);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			if(previousNumExecutingTasks!= info.getNumExecutingTasks()){
				previousNumExecutingTasks = info.getNumExecutingTasks();
				LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Number of executing tasks: " + info.getNumExecutingTasks());
			}
		}
		
		windowsTime = System.currentTimeMillis() - previousCheckingPoint;
		numPreviousCompletedTasks = info.getNumExecutedTasks();
		numCompletedTasksPerWindow = numPreviousCompletedTasks;
		previousCheckingPoint = System.currentTimeMillis();

		previousPerformance = (float) (60000 * numCompletedTasksPerWindow / windowsTime);
		currentPerformance = previousPerformance;
		
		LOG.debug("TaskExecutor [" + info.getExecutorId() + "] Number of tasks per window: " + (numCompletedTasksPerWindow) 
				+ " #activeThreads: " + (info.getNumActiveThreads())
				+ " #exectutedTasks: " + info.getNumExecutedTasks() 
				+ " windowsSize (seconds): " 	+ windowsTime / 1000 
				+ " performance (tasks/minute) : " + currentPerformance
				+ " #executing tasks: " + info.getNumExecutingTasks() 
				+ " #specifiedNumber: " + info.getNumSpecifiedTasks() 
				);
		
		boolean direction = false;
		int step = 1;
		int windowsRatio = 3;
		/* Wait for every thread completed */
		while (info.getNumRunningThreads() > 0) {
			if (info.isQueueEmpty())
				step = 0;
			if (direction) {
				if (info.getNumSpecifiedTasks() > step) {
					info.setNumSpecifiedTasks(info.getNumSpecifiedTasks() - step);
				} else {
					direction = !direction;
				}

			} else {
				info.setNumSpecifiedTasks(info.getNumSpecifiedTasks() + step);
			}
			
			while ((info.getNumExecutedTasks() - numPreviousCompletedTasks) < windowsRatio * info.getNumSpecifiedTasks()) {
				try {
					Thread.sleep(1000);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
								
				if(previousNumExecutingTasks!= info.getNumExecutingTasks()){
					previousNumExecutingTasks = info.getNumExecutingTasks();
					LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + info.getHostname() 
					+ "] Number of running tasks: " + info.getNumExecutingTasks());
				}
				if (info.getNumRunningThreads() == 0)
					break;
			}
			numCompletedTasksPerWindow = info.getNumExecutedTasks() - numPreviousCompletedTasks;
			numPreviousCompletedTasks = info.getNumExecutedTasks();

			windowsTime = System.currentTimeMillis() - previousCheckingPoint;
			previousCheckingPoint = System.currentTimeMillis();

			currentPerformance = (float)(60000 * numCompletedTasksPerWindow / windowsTime);

			if (currentPerformance < previousPerformance) {
				direction = !direction;
				//windowsRatio +=2;
			}

			previousPerformance = currentPerformance;
			/*
			 * wait until all threads exit try { Thread.sleep(windowsTime); } catch (InterruptedException e) { // TODO Auto-generated catch block e.printStackTrace(); } if((info.getNumExecutedTasks()
			 * - numPreviousCompletedTasks) < numCompletedTasksPerWindow){ nagative = !nagative; } numCompletedTasksPerWindow = info.getNumExecutedTasks() - numPreviousCompletedTasks;
			 * numPreviousCompletedTasks = info.getNumExecutedTasks();
			 */

						
			
			LOG.debug("TaskExecutor [" + info.getExecutorId() +"][" + info.getHostname()
					+ "] #tasks/window: " + (numCompletedTasksPerWindow)
					+ " #specifiedNumber: " + info.getNumSpecifiedTasks()
					+ " #activeThreads: " + (info.getNumActiveThreads())
					+ " #running tasks: " + info.getNumExecutingTasks()
					+ " performance (tasks/minute) : " + currentPerformance
					+ " windowsSize (seconds): " 	+ windowsTime / 1000
					+ " #exectutedTasks: " + info.getNumExecutedTasks() 				 	
					 
					);			
			
			
		}
		zkServer.setTimeComplete(zkServer.getSystemTime());
		zkServer.setNumOfProcessedTasks(info.getExecutorId(), info.getNumExecutedTasks());
		zkServer.setStopRequest(true);

		systemCurrentTime = zkServer.getSystemTime();
		/* Display list of files and directories for debugging */
		// curDir = new File(".");
		// getAllFiles(curDir);

		LOG.all("TaskExecutor [" + info.getExecutorId() + "] have completed " + info.getNumExecutedTasks() + " tasks");
		systemCurrentTime = zkServer.getSystemTime();

		info.setEndingTime(systemCurrentTime);
		info.setExecutionTime(info.getEndingTime() - info.getFirstMessageTime());
		info.setNumOfPolls(numOfPolls);
		info.setPollingRate((info.getNumExecutedTasks() / (info.getExecutionTime() / 1000)));
		LOG.debug(info.toString());
		database.insertExecutorInfoToDatabase(info);

		zkServer.setResultsExe(info);
		zkServer.setPerformanceExe(info.getExecutorId(), logs);
		zkServer.close();

		LOG.info(database.toString());
		LOG.all("TaskExecutor [" + info.getExecutorId() + "] exits");
	}

	protected class ThreadTaskExecutor implements Runnable {
		private int id;

		public ThreadTaskExecutor(int id) {
			// TODO Auto-generated constructor stub
			this.id = id;
		}

		@Override
		public void run() {
			// TODO Auto-generated method stub
			
			while (info.getNumSpecifiedTasks() <= id) {
				try {
					Thread.sleep(100);
				} catch (InterruptedException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}	

				if (info.isQueueEmpty()) {
					LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + id + "] queue empty");
					break;
				}
			}

			
			LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + id + "] starts polling and processing jobs got from the job queue");
			/*
			 * Construct zookeeper server which is used to inform number of completed tasks to MOHA Client
			 */

			String appType = System.getenv(MOHA_Properties.APP_TYPE);
			String zookeeperConnect = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_CONNECT);
			String bootstrapServer = System.getenv(MOHA_Properties.KAFKA_ZOOKEEPER_BOOTSTRAP_SERVER);
			String queueType = System.getenv(MOHA_Properties.CONF_QUEUE_TYPE);
			MOHA_Queue jobQueue;
			if (queueType.equals("kafka")) {
				jobQueue = new MOHA_Queue(zookeeperConnect, bootstrapServer, info.getQueueName());
			} else {
				jobQueue = new MOHA_Queue(System.getenv(MOHA_Properties.CONF_ACTIVEMQ_SERVER), info.getQueueName());
			}

			info.setNumRunningThreads(info.getNumRunningThreads() + 1);

			/* Subscribe to the job queue to be allowed polling task input files */
			jobQueue.consumerInit();

			/* Polling first jobs from job queue */

			boolean isProcessingSuccess = false;
			boolean isActive = false;
			int num = 0;
			while (info.getNumRunningThreads() > 0) {
				/* Polling tasks from job queue */
				while (info.getNumSpecifiedTasks() <= id) {
					try {
						Thread.sleep(100);
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					if (isActive) {
						isActive = false;
						info.setNumActiveThreads(info.getNumActiveThreads() - 1);
					}

					if (info.isQueueEmpty()) {
						LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + id + "] queue empty");
						break;
					}
				}
				if ((!isActive) && (!info.isQueueEmpty())) {
					isActive = true;
					info.setNumActiveThreads(info.getNumActiveThreads() + 1);
				}
				info.setNumOfPolls(info.getNumOfPolls() + 1);
				if (jobQueue.isKafka()) {
					/*
					 * if(numOfCommands <1000000){ num = 1; }else{ num = 0; }
					 */
					ConsumerRecords<String, String> records = jobQueue.poll(100);
					num = records.count();

					if (appType.equals("S")) {
						for (ConsumerRecord<String, String> record : records) {
							isProcessingSuccess = executeTask(record.value());
						}
					} else {
						for (ConsumerRecord<String, String> record : records) {
							// LOG.all("TaskExecutor [" + info.getExecutorId() + "]Received: " + record.value());
						}
					}

				} else {
					Message msg = jobQueue.activeMQPoll(1000);
					if (msg != null) {
						if (msg instanceof TextMessage) {
							TextMessage textMessage = (TextMessage) msg;
							String text;
							try {
								text = textMessage.getText();
								int temp;
								if (appType.equals("S")) {
									//LOG.all("TaskExecutor [" + info.getExecutorId() + "][" + id + "]1 #runningTasks: " + info.getNumExecutingTasks());
									temp = info.getNumExecutingTasks();
									info.setNumExecutingTasks(temp + 1);

									isProcessingSuccess = executeTask(text);
									//LOG.all("TaskExecutor [" + info.getExecutorId() + "][" + id + "]2 #runningTasks: " + info.getNumExecutingTasks());
									temp = info.getNumExecutingTasks();
									info.setNumExecutingTasks(temp - 1);
								}

								 //LOG.all("TaskExecutor [" + info.getExecutorId() + "][" + id + "]Received_: " + text);
							} catch (JMSException e) {
								// TODO Auto-generated catch block
								e.printStackTrace();
							}
						} else {
							if (appType.equals("S")) {

								info.setNumExecutingTasks(info.getNumExecutingTasks() + 1);

								isProcessingSuccess = executeTask(msg.toString());

								info.setNumExecutingTasks(info.getNumExecutingTasks() - 1);
							}
							 LOG.all("TaskExecutor [" + info.getExecutorId() + "][" + id + "]Received: " + msg);
						}
						num = 1;
						info.setQueueEmpty(false);
						info.setNumQueueFail(0);
					} else {
						try {
							Thread.sleep(1000);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						num = 0;
						LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + id + "] could not get any message");
						info.setQueueEmpty(true);
						info.setNumQueueFail(info.getNumQueueFail() + 1);
						if (info.getNumQueueFail() > 5) {
							isProcessingSuccess = false;
							break;
						} else {
							isProcessingSuccess = true;
						}
					}
				}
				info.setNumExecutedTasks(info.getNumExecutedTasks() + num);
				if (!isProcessingSuccess) {
					LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + id + "] fail to execture tasks");
					break;
				}
			}
			LOG.debug("TaskExecutor [" + info.getExecutorId() + "][" + id + "] leaves");
			info.setNumRunningThreads(info.getNumRunningThreads() - 1);
			if (isActive) {
				isActive = false;
				info.setNumActiveThreads(info.getNumActiveThreads() - 1);
			}
		}

	}

	private static void getListOfFiles(int executorId, File curDir) {

		File[] filesList = curDir.listFiles();

		for (File f : filesList) {
			if (f.isDirectory()) {
				LOG.debug("TaskExecutor [" + executorId + "] Directory " + f.getName());
				getListOfFiles(executorId, f);
			}
			if (f.isFile()) {
				LOG.debug("TaskExecutor [" + executorId + "] Files: " + curDir.getPath() + "/" + f.getName());
			}
		}

	}

	/**
	 * @return The line number of the code that ran this method
	 * @author Brian_Entei
	 */
	public static int getLineNumber() {
		return ___8drrd3148796d_Xaf();
	}

	/**
	 * This methods name is ridiculous on purpose to prevent any other method names in the stack trace from potentially matching this one.
	 * 
	 * @return The line number of the code that called the method that called this method(Should only be called by getLineNumber()).
	 * @author Brian_Entei
	 */
	private static int ___8drrd3148796d_Xaf() {
		boolean thisOne = false;
		int thisOneCountDown = 1;
		StackTraceElement[] elements = Thread.currentThread().getStackTrace();
		for (StackTraceElement element : elements) {
			String methodName = element.getMethodName();
			int lineNum = element.getLineNumber();
			if (thisOne && (thisOneCountDown == 0)) {
				return lineNum;
			} else if (thisOne) {
				thisOneCountDown--;
			}
			if (methodName.equals("___8drrd3148796d_Xaf")) {
				thisOne = true;
			}
		}
		return -1;
	}
}

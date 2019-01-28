package org.kisti.moha;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CountDownLatch;

import org.apache.zookeeper.CreateMode;
import org.apache.zookeeper.KeeperException;
import org.apache.zookeeper.WatchedEvent;
import org.apache.zookeeper.Watcher;
import org.apache.zookeeper.Watcher.Event.KeeperState;
import org.apache.zookeeper.ZooDefs.Ids;
import org.apache.zookeeper.ZooKeeper;
import org.apache.zookeeper.data.Stat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MOHA_Zookeeper {
	private static final Logger LOG = LoggerFactory.getLogger(MOHA_Zookeeper.class);

	private String clazz;
	private ZooKeeper zkserver;
	private Dirs dirs;

	public MOHA_Zookeeper(Class<?> clazz, String root) {
		this(root);
		setClazz(clazz.getSimpleName());
	}

	public MOHA_Zookeeper(Class<?> clazz, String root, String parent) {
		this(root, parent);
		setClazz(clazz.getSimpleName());
	}

	public MOHA_Zookeeper(String root) {
		this();
		dirs = new Dirs(root);

	}

	public MOHA_Zookeeper(String root, String parent) {
		this();
		dirs = new Dirs(root, parent);

	}

	public String getRoot() {
		return dirs.getRoot();
	}

	public MOHA_Zookeeper() {
		final CountDownLatch connSignal = new CountDownLatch(0);
		LOG.info("Connecting to the zookeeper server");
		try {
			String zookeeperServer = System.getenv().get(MOHA_Properties.CONF_ZOOKEEPER_SERVER);
			System.out.println("MOHA_Zookeeper: zookeeperServer = " + zookeeperServer);
			// zkserver = new ZooKeeper(zookeeperServer, 30000, new Watcher() {
			zkserver = new ZooKeeper("localhost", 30000, new Watcher() {
				@Override
				public void process(WatchedEvent event) {
					// TODO Auto-generated method stub
					if (event.getState() == KeeperState.SyncConnected) {
						connSignal.countDown();
					}
				}
			});
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			connSignal.await();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void createZooKafka() {
		String rootDirs = "/" + MOHA_Properties.ZOOKEEPER_ROOT;
		LOG.info("createRoot:" + rootDirs);
		if (zkserver != null) {
			try {

				Stat s = zkserver.exists(rootDirs, false);
				if (s == null) {
					zkserver.create(rootDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}

		}
	};

	public void createZooMOHA() {
		String rootDirs = "/" + MOHA_Properties.ZOOKEEPER_ROOT;
		LOG.info("createRoot:" + rootDirs);
		if (zkserver != null) {
			try {

				Stat s = zkserver.exists(rootDirs, false);
				if (s == null) {
					zkserver.create(rootDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}

		}
	};

	public void createRoot() {
		String rootDirs = dirs.getRoot();
		LOG.info("createRoot:" + rootDirs);
		if (zkserver != null) {

			try {

				Stat s = zkserver.exists(rootDirs, false);
				if (s == null) {
					zkserver.create(rootDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}

		}
	};

	public void createDirs(String child) {
		String path_ = dirs.getPath(child);
		LOG.info("createDirs:" + path_);
		if (zkserver != null) {

			try {

				Stat s = zkserver.exists(path_, false);
				if (s == null) {
					zkserver.create(path_, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}

		}
	};

	public void createDirsFull(String path) {
		String path_ = path;
		//LOG.info("createDirs:" + path_);
		if (zkserver != null) {

			try {

				Stat s = zkserver.exists(path_, false);
				if (s == null) {
					zkserver.create(path_, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}

		}
	};

	public void create(List<String> iDirs) {

		this.createRoot();
		if (zkserver != null) {
			for (String subDirs : iDirs) {
				try {

					String path = dirs.getPath(subDirs);

					Stat s = zkserver.exists(path, false);
					if (s == null) {
						zkserver.create(path, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);
					}

				} catch (KeeperException e) {
					System.out.println("Keeper exception when instantiating queue: " + e.toString());
				} catch (InterruptedException e) {
					System.out.println("Interrupted exception");
				}
			}

		}
	};

	public void setLogs(int containerId, String msg) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOGS);
		String logs;

		try {
			if (zkserver.exists(logDir, false) == null) {
				LOG.info("Error: There are no node: " + logDir);
				List<String> ids;
				ids = zkserver.getChildren("/", false);
				for (String id : ids) {
					LOG.info(id);
				}
				return;
			}
			logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOGS, String.valueOf(containerId));
			createDirsFull(logDir);

			logs = new String(zkserver.getData(logDir, false, null)) + MOHA_Common.convertLongToDate(System.currentTimeMillis()) + " MOHA_LOG " + getClazz()
					+ " : " + msg;
			zkserver.setData(logDir, logs.getBytes(), -1);
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void setResultsApp(MOHA_Info appInfo) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_APP_LOG);
		String logs;
		// "\n" + "appId executorMemory numExecutors numPartitions startingTime
		// initTime makespan numCommands command \n"
		logs = appInfo.getAppId() + " " + appInfo.getExecutorMemory() + " " + appInfo.getNumExecutors() + " " + appInfo.getNumPartitions() + " "
				+ MOHA_Common.convertLongToDate(System.currentTimeMillis()) + " " + appInfo.getInitTime() + " " + appInfo.getMakespan() + " "
				+ appInfo.getNumCommands() + " " + appInfo.getCommand() + "\n";

		try {
			if (zkserver.exists(logDir, false) == null) {
				LOG.info("Error: There are no node: " + logDir);
				List<String> ids;
				ids = zkserver.getChildren("/", false);
				for (String id : ids) {
					LOG.info(id);
				}
				return;
			}
			zkserver.setData(logDir, logs.getBytes(), -1);
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public String getResultsApp() {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_APP_LOG);

		try {
			if (zkserver.exists(logDir, false) == null)
				return "";
			String logs = new String(zkserver.getData(logDir, false, null));
			zkserver.setData(logDir, "".getBytes(), -1);
			return logs;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return "";

	}

	public void setResultsExe(MOHA_ExecutorInfo eInfo) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG);

		String logs;
		/*appid exeid containerId hostname date time numPolls numTasks rate exeTime rate launchTime WaitingTime FirstM EndingTime*/
		logs = eInfo.getAppId() + " " + eInfo.getExecutorId() + " " + eInfo.getContainerId() + " " + eInfo.getHostname() + " "
				+ MOHA_Common.convertLongToDate(System.currentTimeMillis())+ " : " + eInfo.getNumOfPolls()  + " " + eInfo.getNumExecutedTasks() + " " + eInfo.getExecutionTime() + " " + eInfo.getPollingRate() + " "
				+ eInfo.getLaunchedTime() + " " + eInfo.getWaitingTime()   + " " + eInfo.getFirstMessageTime() + " "
				+ eInfo.getEndingTime() + "\n";
		try {
			if (zkserver.exists(logDir, false) == null) {
				LOG.info("Error: There are no node: " + logDir);
				List<String> ids;
				ids = zkserver.getChildren("/", false);
				for (String id : ids) {
					LOG.info(id);
				}
				return;
			}
			logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG, String.valueOf(eInfo.getExecutorId()));
			createDirsFull(logDir);
			zkserver.setData(logDir, logs.getBytes(), -1);

		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public void setPerformanceExe(int exeId, String logs) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG);

		try {
			if (zkserver.exists(logDir, false) == null) {
				LOG.info("Error: There are no node: " + logDir);
				List<String> ids;
				ids = zkserver.getChildren("/", false);
				for (String id : ids) {
					LOG.info(id);
				}
				return;
			}
			logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG, String.valueOf(exeId));
			createDirsFull(logDir);
			zkserver.setData(logDir, logs.getBytes(), -1);

		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public String getResultsExe() {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG);
		String logs = "";
		String temp = "";

		try {
			if (zkserver.exists(logDir, false) == null)
				return "";

			List<String> ids = zkserver.getChildren(logDir, false);
			//LOG.info("Num of ids is {}, ids = {}", ids.size(), ids.toString());
			for (String id : ids) {

				logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG, id);
				temp = new String(zkserver.getData(logDir, false, null));
				//LOG.info("id = {}, content = {} \n", id, temp);
				logs += temp;
				zkserver.setData(logDir, "".getBytes(), -1);
			}
			return logs;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return "";

	}

	public String getPerformanceExe() {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG);
		String logs = "";
		String temp = "";

		try {
			if (zkserver.exists(logDir, false) == null)
				return "";

			List<String> ids = zkserver.getChildren(logDir, false);
			//LOG.info("Num of ids is {}, ids = {}", ids.size(), ids.toString());
			for (String id : ids) {

				logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG, id);
				temp = new String(zkserver.getData(logDir, false, null));
				//LOG.info("id = {}, content = {} \n", id, temp);
				logs += temp;
				zkserver.setData(logDir, "".getBytes(), -1);
			}
			return logs;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return "";

	}

	public Boolean createExecutorResultsDir(int brokerId) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG);
		try {
			if (zkserver.exists(logDir, false) == null) {
				LOG.info("Error: There are no node: " + logDir);
				List<String> ids;
				ids = zkserver.getChildren("/", false);
				for (String id : ids) {
					LOG.info(id);
				}
				return false;
			}
			logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG, String.valueOf(brokerId));
			createDirsFull(logDir);
			return true;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return false;
	}

	public Boolean createExecutorPerformanceDir(int brokerId) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG);
		try {
			if (zkserver.exists(logDir, false) == null) {
				LOG.info("Error: There are no node: " + logDir);
				List<String> ids;
				ids = zkserver.getChildren("/", false);
				for (String id : ids) {
					LOG.info(id);
				}
				return false;
			}
			logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXE_PERFORMANCE_LOG, String.valueOf(brokerId));
			createDirsFull(logDir);
			return true;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return false;
	}

	public int getNumExecutorsRunning() {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_RESULTS_EXE_LOG);

		try {
			if (zkserver.exists(logDir, false) == null)
				return 0;

			List<String> ids = zkserver.getChildren(logDir, false);
			LOG.info("ids = {}", ids.toString());
			return ids.size();

		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return 0;
	}

	public void setCompletedJobsNum(int brokerId, int completedJobsNum) {

		String idsDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXECUTORS, String.valueOf(brokerId));

		try {

			zkserver.setData(idsDirs, String.valueOf(completedJobsNum).getBytes(), -1);

		} catch (KeeperException e) {
			System.out.println("Keeper exception when instantiating queue: " + e.toString());
		} catch (InterruptedException e) {
			System.out.println("Interrupted exception");
		}

	}

	public int getCompletedJobsNum(int brokerId) {
		String idsDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXECUTORS, String.valueOf(brokerId));
		String logs;
		try {
			logs = new String(zkserver.getData(idsDirs, false, null));
			return Integer.parseInt(logs);
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return 0;
	}

	public int getAllCompletedJobsNum() {

		Dirs root = new Dirs(dirs, MOHA_Properties.ZOOKEEPER_DIR_EXECUTORS);
		int sum = 0;
		List<String> ids;
		try {
			ids = zkserver.getChildren(dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_EXECUTORS), false);
			LOG.info("ids = {}", ids.toString());
			for (String id : ids) {
				String num = new String(zkserver.getData(root.getPath(id), false, null));
				LOG.info("sum = {}", num);
				sum += Integer.parseInt(num);
				LOG.info("TaskExecutor [" + id + "] have completed " + num + " jobs");
			}

		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		LOG.info("Sum = " + String.valueOf(sum));
		return sum;
	}

	public String getLogs() {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOGS);
		String logs = "";

		try {
			if (zkserver.exists(logDir, false) == null)
				return "";

			List<String> ids = zkserver.getChildren(logDir, false);
			//LOG.info("ids = {}", ids.toString());
			for (String id : ids) {
				logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOGS, id);
				logs += new String(zkserver.getData(logDir, false, null));
				//System.out.println(logs);
				zkserver.setData(logDir, "".getBytes(), -1);
			}
			return logs;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return "";

	}

	public void setPollingEnable(boolean enable) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setStatus() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}

					if (enable) {
						// LOG.info("SetData = true dir = " + statusDirs);
						zkserver.setData(requestDirs, "True".getBytes(), -1);
					} else {
						// LOG.info("SetData = false dir = " + statusDirs);
						zkserver.setData(requestDirs, "False".getBytes(), -1);
					}
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public void setSystemTime(long time) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE, "time");

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setStatus() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					String time_ = String.valueOf(time);
					zkserver.setData(requestDirs, time_.getBytes(), -1);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public long getSystemTime() {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE, "time");

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String stInfo = new String(zkserver.getData(requestDirs, false, null));
						return Long.parseLong(stInfo);
					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return System.currentTimeMillis();
	}

	public void setTimeStart(long time) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_TIME_START);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setStatus() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					String stInfo = new String(zkserver.getData(requestDirs, false, null));
					// If no one come
					if (stInfo.length() == 0) {
						String time_ = String.valueOf(time);
						zkserver.setData(requestDirs, time_.getBytes(), -1);
					}
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public long getTimeStart() {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_TIME_START);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String stInfo = new String(zkserver.getData(requestDirs, false, null));
						if (stInfo.length() > 0) {
							return Long.parseLong(stInfo);
						}
					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return 0;
	}

	public void setTimeComplete(long time) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_TIME_COMPLETE);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setStatus() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					String time_ = String.valueOf(time);
					zkserver.setData(requestDirs, time_.getBytes(), -1);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public long getTimeComplete() {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_TIME_COMPLETE);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String stInfo = new String(zkserver.getData(requestDirs, false, null));
						if (stInfo.length() > 0) {
							return Long.parseLong(stInfo);
						}

					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return 0;
	}

	public void setNumOfProcessedTasks(int id, int num) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_NUM_PROCESSED_TASKS, String.valueOf(id));

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setStatus() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					String time_ = String.valueOf(num);
					zkserver.setData(requestDirs, time_.getBytes(), -1);
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public int getNumOfProcessedTasks() {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_NUM_PROCESSED_TASKS);
		int num = 0;

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {

						List<String> ids = zkserver.getChildren(requestDirs, false);
						// LOG.info("ids = {}", ids.toString());
						for (String id : ids) {
							String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_NUM_PROCESSED_TASKS, id);
							String stInfo = new String(zkserver.getData(logDir, false, null));
							if (stInfo.length() > 0) {
								num += Integer.parseInt(stInfo);
							}
						}

					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return num;
	}

	public boolean getPollingEnable() {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String stInfo = new String(zkserver.getData(requestDirs, false, null));
						return Boolean.parseBoolean(stInfo);
					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return false;

	}

	public void setTimming(long time) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setTimming() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for request");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					LOG.info("Set time");
					zkserver.setData(requestDirs, String.valueOf(time).getBytes(), -1);

				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public long getTimming() {

		// LOG.info("Checking time base");

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_POLLING_ENABLE);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String rqInfo = new String(zkserver.getData(requestDirs, false, null));

						LOG.info("getTimming = {} time ------------------------- = {}", requestDirs, rqInfo);

						return Long.parseLong(rqInfo);

					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return 0;

	}

	public void setStopRequest(Boolean stop) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_STOP);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setRequests() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for request");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}

					if (stop) {
						zkserver.setData(requestDirs, "True".getBytes(), -1);
						//LOG.info("set stop requrest true");
					} else {
						zkserver.setData(requestDirs, "False".getBytes(), -1);
						//LOG.info("set stop requrest false");
					}
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}
	
	public void setLocalResource(String name) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOCAL_RESOURCE);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setRequests() fail");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for request");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					zkserver.setData(requestDirs, name.getBytes(), -1);
					
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}
	
	public String getLocalResource() {

		LOG.info("Get local resource");

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOCAL_RESOURCE);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String localResource = new String(zkserver.getData(requestDirs, false, null));

						//LOG.info("requestDirs = {} rqInfo ------------------------- = {}", requestDirs, rqInfo);

						return localResource;

					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return "";

	}

	public boolean getStopRequest() {

		//LOG.info("Checking request from MOHA manager");

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_STOP);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String rqInfo = new String(zkserver.getData(requestDirs, false, null));

						//LOG.info("requestDirs = {} rqInfo ------------------------- = {}", requestDirs, rqInfo);
						//LOG.info("true or false:" + rqInfo);
						return Boolean.parseBoolean(rqInfo);

					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		//LOG.info("true by default");
		return true;

	}

	public void setManagerRunning(Boolean isRunning) {

		String rootDirs = dirs.getRoot();
		String statusDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_STATUS);
		// LOG.info("setStatus.statusDirs = {}",statusDirs);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("Zookeeper Broker is not running yet. Root directory: " + rootDirs);
					LOG.info("setStatus() fail");
					return;
				} else {
					Stat s = zkserver.exists(statusDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(statusDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}

					if (isRunning) {
						// LOG.info("SetData = true dir = " + statusDirs);
						zkserver.setData(statusDirs, "True".getBytes(), -1);
					} else {
						// LOG.info("SetData = false dir = " + statusDirs);
						zkserver.setData(statusDirs, "False".getBytes(), -1);
					}
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public boolean isManagerRunning() {

		// LOG.info("Checking request from MOHA manager");
		String rootDirs = dirs.getRoot();
		String statusDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_STATUS);
		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(statusDirs, false) != null) {
						String stInfo = new String(zkserver.getData(statusDirs, false, null));
						return Boolean.parseBoolean(stInfo);
					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return true;

	}

	// public String getBootstrapServers() {
	// return
	// "hdp01.kisti.re.kr:9092,hdp02.kisti.re.kr:9092,hdp03.kisti.re.kr:9092,hdp04.kisti.re.kr:9092,hdp05.kisti.re.kr:9092,hdp06.kisti.re.kr:9092";
	// }

	public String getBootstrapServers() {

		// Show broker server information
		String zookeeperConnect = "localhost:9092";
		StringBuilder command = new StringBuilder();
		Dirs root = new Dirs(dirs, "brokers", "ids");

		List<String> ids;
		try {
			if (zkserver.exists(root.getRoot(), false) == null) {
				return null;
			}
			ids = zkserver.getChildren(root.getRoot(), false);
			LOG.info("ids = {}", ids.toString());
			for (String id : ids) {
				String brokerInfo = new String(zkserver.getData(root.getPath(id), false, null));
				LOG.info("server = {}", brokerInfo.substring(brokerInfo.lastIndexOf("[") + 14, brokerInfo.lastIndexOf("]") - 1));
				command.append(brokerInfo.substring(brokerInfo.lastIndexOf("[") + 14, brokerInfo.lastIndexOf("]") - 1)).append(",");
			}
			command.deleteCharAt(command.length() - 1);

			zookeeperConnect = command.toString();
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		LOG.info("getBootstrapServers.zookeeperConnect = " + zookeeperConnect);
		return zookeeperConnect;
	}

	public boolean isKafkaDebugServiceAvailable() {
		if (zkserver != null) {

			// Dirs brokersExist = new Dirs("brokers");
			Dirs idsExist = new Dirs("brokers", "ids");

			try {
				if (zkserver.exists(idsExist.getRoot(), false) == null) {
					return false;
				}
				List<String> ids = zkserver.getChildren(idsExist.getRoot(), false);
				if (ids.size() > 0) {
					return true;
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return false;
	}

	public boolean isBrokerStated() {
		if (zkserver != null) {

			try {
				List<String> ids = zkserver.getChildren(dirs.getPath("brokers", "ids"), false);
				if (ids.size() > 0) {
					return true;
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return false;
	}

	public void close() {
		try {
			zkserver.close();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void delete() {
		String rootDirs = dirs.getRoot();
		try {
			zkserver.delete(rootDirs, zkserver.exists(rootDirs, true).getVersion());
		} catch (InterruptedException | KeeperException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void delete(String dir) {
		try {
			if (zkserver.exists(dir, false) != null) {
				List<String> ids = zkserver.getChildren(dir, false);
				if (ids.size() > 0) {
					for (String id : ids) {
						delete(dir + "/" + id);
					}
				}
				zkserver.delete(dir, zkserver.exists(dir, true).getVersion());
			}
		} catch (KeeperException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public String getClazz() {
		return clazz;
	}

	public void setClazz(String clazz) {
		this.clazz = clazz;
	}

	public boolean exist() {
		// TODO Auto-generated method stub
		String rootDirs = dirs.getRoot();
		try {
			if (zkserver.exists(rootDirs, false) != null) {

				return true;
			}
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return false;
	}

	private class Dirs {
		String root;

		public Dirs(String root) {
			super();
			this.root = "/" + root;
		}

		public Dirs(Dirs root, String child) {
			super();
			this.root = root.getRoot() + "/" + child;
		}

		public Dirs(String child1, String child2) {
			super();
			this.root = "/" + child1 + "/" + child2;
		}

		public Dirs(Dirs root, String child1, String child2) {
			super();
			this.root = root.getRoot() + "/" + child1 + "/" + child2;
		}

		public String getPath(String child) {
			return root + "/" + child;
		}

		public String getPath(String child1, String child2) {
			return root + "/" + child1 + "/" + child2;
		}

		public String getRoot() {
			return root;
		}
	}
}

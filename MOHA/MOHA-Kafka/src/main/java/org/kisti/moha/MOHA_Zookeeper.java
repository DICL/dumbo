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
	public String getRoot(){
		return dirs.getRoot();
	}

	public MOHA_Zookeeper() {
		final CountDownLatch connSignal = new CountDownLatch(0);
		LOG.info("Connecting to the zookeeper server");
		try {
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

	public void setLogs(String msg) {

		String logDir = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_LOGS);
		String logs;
		String preLogs;

		preLogs = getLogs();

		if (preLogs.length() > 0) {
			logs = preLogs + "\n" + MOHA_Common.convertLongToDate(System.currentTimeMillis()) + " MOHA_LOG "
					+ getClazz() + " : " + msg;
		} else {
			logs = MOHA_Common.convertLongToDate(System.currentTimeMillis()) + " MOHA_LOG " + getClazz() + " : " + msg;
		}
		try {
			zkserver.setData(logDir, logs.getBytes(), -1);
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

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

		try {
			String logs = new String(zkserver.getData(logDir, false, null));
			zkserver.setData(logDir, "".getBytes(), -1);
			return logs;
		} catch (KeeperException | InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return "";

	}

	public void setReadyfor(int brokerid, long value) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_TIME + "/" + brokerid);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("broker is not running yet");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for request");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					LOG.info("setReadyfor {} value {}", requestDirs, value);
					zkserver.setData(requestDirs, String.valueOf(value).getBytes(), -1);

				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public long getReadyfor(int brokerid) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_TIME + "/" + brokerid);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String rqInfo = new String(zkserver.getData(requestDirs, false, null));

						LOG.info("getReadyfor = {} time ------------------------- = {}", requestDirs, rqInfo);

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

	public void setTimming(long time) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_TIME);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("broker is not running yet");
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
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_TIME);

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

	public void setRequests(Boolean stop) {

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_START_STOP);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("broker is not running yet");
					return;
				} else {
					Stat s = zkserver.exists(requestDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for request");
						zkserver.create(requestDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}
					LOG.info("Making the request");
					if (stop) {
						zkserver.setData(requestDirs, "true".getBytes(), -1);
					} else {
						zkserver.setData(requestDirs, "false".getBytes(), -1);
					}
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public boolean getRequests() {

		LOG.info("Checking request from MOHA manager");

		String rootDirs = dirs.getRoot();
		String requestDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_REQUEST_START_STOP);

		if (zkserver != null) {
			try {
				if (zkserver.exists(rootDirs, false) != null) {
					if (zkserver.exists(requestDirs, false) != null) {
						String rqInfo = new String(zkserver.getData(requestDirs, false, null));

						LOG.info("requestDirs = {} rqInfo ------------------------- = {}", requestDirs, rqInfo);

						return Boolean.parseBoolean(rqInfo);

					}
				}
			} catch (KeeperException | InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}

		return true;

	}

	public void setStatus(Boolean isRunning) {

		String rootDirs = dirs.getRoot();
		String statusDirs = dirs.getPath(MOHA_Properties.ZOOKEEPER_DIR_STATUS);
		// LOG.info("setStatus.statusDirs = {}",statusDirs);

		if (zkserver != null) {
			try {
				Stat root = zkserver.exists(rootDirs, false);
				if (root == null) {
					LOG.info("broker is not running yet");
					return;
				} else {
					Stat s = zkserver.exists(statusDirs, false);
					if (s == null) {
						LOG.info("Creating a znode for status");
						zkserver.create(statusDirs, new byte[0], Ids.OPEN_ACL_UNSAFE, CreateMode.PERSISTENT);

					}

					if (isRunning) {
						// LOG.info("SetData = true dir = " + statusDirs);
						zkserver.setData(statusDirs, "true".getBytes(), -1);
					} else {
						// LOG.info("SetData = false dir = " + statusDirs);
						zkserver.setData(statusDirs, "false".getBytes(), -1);
					}
				}

			} catch (KeeperException e) {
				System.out.println("Keeper exception when instantiating queue: " + e.toString());
			} catch (InterruptedException e) {
				System.out.println("Interrupted exception");
			}
		}

	}

	public boolean getStatus() {

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
			ids = zkserver.getChildren(root.getRoot(), false);
			LOG.info("ids = {}", ids.toString());
			for (String id : ids) {
				String brokerInfo = new String(zkserver.getData(root.getPath(id), false, null));
				LOG.info("server = {}",
						brokerInfo.substring(brokerInfo.lastIndexOf("[") + 14, brokerInfo.lastIndexOf("]") - 1));
				command.append(brokerInfo.substring(brokerInfo.lastIndexOf("[") + 14, brokerInfo.lastIndexOf("]") - 1))
						.append(",");
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

			Dirs root = new Dirs("brokers", "ids");

			try {
				List<String> ids = zkserver.getChildren(root.getRoot(), false);
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

	public boolean isRunning() {
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

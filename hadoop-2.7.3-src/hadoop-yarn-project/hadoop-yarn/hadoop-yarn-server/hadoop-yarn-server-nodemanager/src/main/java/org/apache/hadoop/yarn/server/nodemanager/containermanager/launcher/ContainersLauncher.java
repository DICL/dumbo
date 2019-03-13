/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.hadoop.yarn.server.nodemanager.containermanager.launcher;

import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileContext;
import org.apache.hadoop.fs.UnsupportedFileSystemException;
import org.apache.hadoop.service.AbstractService;
import org.apache.hadoop.yarn.api.records.ContainerId;
import org.apache.hadoop.yarn.event.Dispatcher;
import org.apache.hadoop.yarn.event.EventHandler;
import org.apache.hadoop.yarn.exceptions.YarnRuntimeException;
import org.apache.hadoop.yarn.server.nodemanager.ContainerExecutor;
import org.apache.hadoop.yarn.server.nodemanager.Context;
import org.apache.hadoop.yarn.server.nodemanager.LocalDirsHandlerService;
import org.apache.hadoop.yarn.server.nodemanager.containermanager.ContainerManagerImpl;
import org.apache.hadoop.yarn.server.nodemanager.containermanager.application.Application;
import org.apache.hadoop.yarn.server.nodemanager.containermanager.container.Container;
import org.apache.hadoop.yarn.server.nodemanager.containermanager.localizer.ResourceLocalizationService;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.util.concurrent.ThreadFactoryBuilder;

/**
 * The launcher for the containers. This service should be started only after
 * the {@link ResourceLocalizationService} is started as it depends on creation
 * of system directories on the local file-system.
 * 
 */
public class ContainersLauncher extends AbstractService implements EventHandler<ContainersLauncherEvent> {

	private static final Log LOG = LogFactory.getLog(ContainersLauncher.class);

	private final Context context;
	private final ContainerExecutor exec;
	private final Dispatcher dispatcher;
	private final ContainerManagerImpl containerManager;

	/********************************************************************
	 * SUBCLASS
	 ********************************************************************/
	class AppMaster {
		private int max_num_of_container;
		private int running_num_of_mr;
		private int running_num_of_all_yarnchild;
		private int allocationMb;
		private int K;
		private int num_map_task;

		public AppMaster()
		{
			max_num_of_container = 0;
			allocationMb = 0;

			K = 0;
			running_num_of_mr = 0;
			running_num_of_all_yarnchild = 0;
			num_map_task = 0;
		}

		public void setMaxSlot(int m) {
			this.max_num_of_container = m;
		}

		public void setK(int k) { this.K=k; }
		public int getK() { return K; }

		public int getMaxSlot() {
			return this.max_num_of_container;
		}

		public void setAllocationMB(int mb) {
			allocationMb = mb;
			// FIXME: why it's divided by allocationMb?
			if (this.max_num_of_container != -1)
				this.max_num_of_container = this.max_num_of_container / mb;
		}

		public void newMaster() {
			this.running_num_of_mr++;
		}

		public void delMaster() {
			this.running_num_of_mr--;
		}

		public int getMasterNum() {
			return running_num_of_mr;
		}

		public void newChild() {
			this.running_num_of_all_yarnchild++;
		}

		public void delChild() {
			this.running_num_of_all_yarnchild--;
		}

		public int getChildNum() {
			return running_num_of_all_yarnchild;
		}

		public void setNumMapTask(String mycmd) {
			String s_num_map_task = (String)mycmd.subSequence(mycmd.indexOf("num.map.tasks=")+14, mycmd.indexOf("num.map.tasks=")+20);
			s_num_map_task = s_num_map_task.substring(0, s_num_map_task.indexOf(" "));
			this.num_map_task = Integer.parseInt(s_num_map_task);
		}

		public void setNumMapTask(int num_map_task) {
			this.num_map_task = num_map_task;
		}

		public int getNumMapTask() {
			return this.num_map_task;
		}

	}

	class ChildMaster {
		private int finished;
		private int count;
		private int max_num_yarnchild;
		private int min_num_yarnchild;
		private int running_num_yarnchild;
		final static int threshold_num_for_getting_P = 1;
		final static int latest_n = 10;
		int cur_n;
		private HashMap<Integer, Long> running_time_of_a_task; 
		long whole_running_time;
		long[] n_running_time;

		private boolean use_multithread;
		private int num_multithread;

		ChildMaster() {
			count = 0;
			finished = 0;
			max_num_yarnchild = 0;
			min_num_yarnchild = 0;
			running_num_yarnchild = 0;
			whole_running_time = 0;

			cur_n = 0;
			n_running_time = new long[latest_n];
			for (int i=0; i<latest_n; i++) {
				n_running_time[i] = -1;
			}

			running_time_of_a_task = new HashMap<Integer, Long>();

			use_multithread = false;
			num_multithread = 1;
		}
		public void clear() {
			count = 0;
			finished = 0;
			running_num_yarnchild = 0;
			whole_running_time = 0;

			running_time_of_a_task = new HashMap<Integer, Long>();

			use_multithread = false;
			num_multithread = 1;
		}

		public void willLaunch(ContainerId containerId) {
			running_num_yarnchild++;
			count++;

			@SuppressWarnings("deprecation")
				int cid = containerId.getId();
			running_time_of_a_task.put(cid, System.currentTimeMillis());
		}

		public void willCleanup(ContainerId containerId) {
			running_num_yarnchild--;
			finished++;

			@SuppressWarnings("deprecation")
				int cid = containerId.getId();
			long duration = System.currentTimeMillis() - running_time_of_a_task.getOrDefault(cid, 0L);
			whole_running_time += duration;

			n_running_time[cur_n++] = duration;
			if (cur_n >= latest_n) cur_n = 0;
		}

		public void setMaxChild(int n) {
			this.max_num_yarnchild = n;
		}

		public int getMaxChild() {
			return this.max_num_yarnchild;
		}

		public void setMinChild(int n) {
			this.min_num_yarnchild = n;
		}

		public void setUseMultithread(boolean b) {
			this.use_multithread = b;
		}

		public void setNumMultithread(int n) {
			this.num_multithread = n;
		}

		public int getMinChild() {
			return this.min_num_yarnchild;
		}

		public int getRunning() {
			return running_num_yarnchild;
		}

		public int getFinished() {
			return finished;
		}

		public int getThreshold() {
			return threshold_num_for_getting_P;
		}

		public long getRunningTime() {
			return whole_running_time;
		}

		public int getCount() {
			return count;
		}

		public boolean getUseMultithread() {
			return use_multithread;
		}

		public int getNumMultithread() {
			return num_multithread;
		}
	}

	/********************************************************************
	 * CLASS MEMBERS
	 ********************************************************************/
	enum TaskType {
		MAP, REDUCE, UNKNOWN
	}

	AppMaster AM;
	ChildMaster CPU, GPU;

	// for socket to debug
	String debug_listener_address = "";
	String expr_title = "";
	int debug_listener_port;
	boolean bUse_debug_listener = false;

	// CPU or GPU mode
	boolean bOnlyCPU = false;
	boolean bMustGPU = false;

	// Records per kernel launch (Default: 10k)
	int R = 10240;

	// Using MPS?
	boolean bUseMps = false;

	// Some configurations for hybrid execution
	public float cpugpu_proportion = 1;
	public int num_of_nodes = 1;
	public boolean bUse_dynamic_scheduler = false;
	public float upper_threshold = 90;
	public float gpu_threshold = 0.7f;
	Thread gpumonitorthread;
	GPUMonitor gpumonitor;

	LOAD_BALANCING_POLICY lbp;

	enum LOAD_BALANCING_POLICY {
		auto, threshold;
	}

	private LocalDirsHandlerService dirsHandler;
	@VisibleForTesting
		public ExecutorService containerLauncher = Executors
		.newCachedThreadPool(new ThreadFactoryBuilder().setNameFormat("ContainersLauncher #%d").build());
	@VisibleForTesting
		public final Map<ContainerId, ContainerLaunch> running = Collections
		.synchronizedMap(new HashMap<ContainerId, ContainerLaunch>());

	// private Configuration conf;
	private Semaphore sem, sem2;

	/********************************************************************
	 * METHODS
	 ********************************************************************/
	public ContainersLauncher(Context context, Dispatcher dispatcher, ContainerExecutor exec,
			LocalDirsHandlerService dirsHandler, ContainerManagerImpl containerManager) {
		super("containers-launcher");
		this.exec = exec;
		this.context = context;
		this.dispatcher = dispatcher;
		this.dirsHandler = dirsHandler;
		this.containerManager = containerManager;

		}

	protected boolean isGpuAvailable(int cur_gpu, int max_gpu) {
		return isGpuAvailable(cur_gpu, max_gpu, false);
	}

	protected boolean isGpuAvailable(int cur_gpu, int max_gpu, boolean isNewMapTask) {
		return isGpuAvailable(cur_gpu, max_gpu, isNewMapTask, false);
	}

	protected boolean isGpuAvailable(int cur_gpu, int max_gpu, boolean isNewMapTask, boolean isMapPhase) {
		boolean ret = false;
		if (bUse_dynamic_scheduler) {

			try {
				sem.acquire(1);
				try {
					ret = gpumonitor.isGpuAvailable(cur_gpu, max_gpu, isNewMapTask);
				} finally {
					sem.release(1);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
		} else {
			if (GPU.getRunning() < GPU.getMaxChild())
				ret = true;
			else
				ret = false;
		}
		return ret;
	}

	@Override
		protected void serviceInit(Configuration conf) throws Exception {
			try {
				// TODO Is this required?
				FileContext.getLocalFSFileContext(conf);
			} catch (UnsupportedFileSystemException e) {
				throw new YarnRuntimeException("Failed to start ContainersLauncher", e);
			}

			AM  = new AppMaster();
			CPU = new ChildMaster();
			GPU = new ChildMaster();

			// yildbs added
			this.upper_threshold = conf.getFloat("myconf.upper.threshold", 96.0f);
			this.gpu_threshold = conf.getFloat("myconf.gpu.threshold", 0.7f);

			this.cpugpu_proportion = conf.getFloat("myconf.cpugpu.proportion", 1f);
			this.num_of_nodes = conf.getInt("myconf.num.nodes", 1);

			this.debug_listener_address = conf.get("myconf.debug.listener.address", "node01");
			this.debug_listener_port = conf.getInt("myconf.debug.listener.port", 51231);
			this.bUse_debug_listener = conf.getBoolean("myconf.debug.listener.use", false);
			this.expr_title = conf.get("myconf.experiment.name", "");

			this.R = conf.getInt("myconf.gpu.r", 10240);

			//int matmul_probing_time_sec = conf.getInt("myconf.gpumonitor.matmul.prbing-time-sec", 1);
			long time_to_wait = 1000 * conf.getLong("myconf.gpumonitor.time-to-wait-sec", 15);
			long time_to_monitor = 1000 * conf.getLong("myconf.gpumonitor.time-to-monitor-sec", 120);
			float matmul_time_threshold = conf.getFloat("myconf.gpumonitor.matmul.time-threshold", 2.0f);
			float matmul_count_threshold_percentage = conf.getFloat("myconf.gpumonitor.matmul.count-threshold", 0.6f);

			int matmul_length = (int) (15 * time_to_monitor / 1000);
			int matmul_count_threshold = (int) (matmul_length * matmul_count_threshold_percentage);

			this.bUse_dynamic_scheduler = conf.getBoolean("myconf.use.dynamic.scheduler", false);
			sem = new Semaphore(1, true);
			sem2 = new Semaphore(1, true);
			gpumonitor = new GPUMonitor(debug_listener_address, debug_listener_port, bUse_debug_listener, upper_threshold, matmul_length, matmul_time_threshold, matmul_count_threshold, sem, sem2);

			boolean max_start_mode = conf.getBoolean("myconf.use.maxstart.scheduler", true);
			boolean bUseNvml = conf.getBoolean("myconf.gpumonitor.use.nvml", true);
			boolean bUseMatmul = conf.getBoolean("myconf.gpumonitor.use.matmul", true);
			boolean bUseOriginalScheduler = conf.getBoolean("myconf.gpumonitor.use.original.scheduler", false);

			gpumonitor.setTimeToWait(time_to_wait);
			gpumonitor.setTimeToMonitor(time_to_monitor);
			gpumonitor.useNvml(bUseNvml);
			gpumonitor.useMatmul(bUseMatmul);
			gpumonitor.useOriginalScheduler(bUseOriginalScheduler);
			gpumonitor.useDynamicScheduler(bUse_dynamic_scheduler);

			gpumonitorthread = new Thread(gpumonitor);
			gpumonitorthread.start();

			String slbp = conf.get("myconf.load.balancing.policy", "threshold");
			if (slbp.startsWith("threshold")) {
				this.lbp = LOAD_BALANCING_POLICY.threshold;
			} else if (slbp.startsWith("auto")) {
				this.lbp = LOAD_BALANCING_POLICY.auto;
			}

			AM.setK(conf.getInt("myconf.num.containers", 1));
			AM.setMaxSlot(conf.getInt("yarn.nodemanager.resource.memory-mb", -1));
			AM.setAllocationMB(conf.getInt("yarn.scheduler.minimum-allocation-mb", 1024));
			GPU.setMaxChild(Math.min(conf.getInt("myconf.num.gpu.yarnchild", 4), AM.getMaxSlot()));

			// hyeonjin added 2017-03-20
			// limit NG to 15 because the number of GPU process on MPS is restricted
			// to 16 and NodeManager uses one for matmul.
			this.bUseMps = conf.getBoolean("myconf.use.mps", false);
			if (bUseMps)
				GPU.setMaxChild(Math.min(GPU.getMaxChild(),  15));

			gpumonitor.setMinGpu(GPU.getMinChild());
			gpumonitor.setMaxGpu(GPU.getMaxChild());

			if (this.bUse_dynamic_scheduler) 
				gpumonitor.SetMaxStartMode(max_start_mode);
			else
				gpumonitor.X = GPU.getMaxChild();

			GPU.setMinChild(conf.getInt("myconf.num.min.gpu.yarnchild", 1));
			if (bUse_debug_listener)
				gpumonitor.setNumMinGpuMapTask(GPU.getMinChild());

			if (GPU.getMaxChild() == 0)
				bOnlyCPU = true;

			CPU.setMaxChild(AM.getMaxSlot() - GPU.getMaxChild());

			// hyeonjin added 2017-12-24
			CPU.setUseMultithread(conf.getBoolean("myconf.use.multithread", false));
			if (CPU.getUseMultithread()) {
				CPU.setNumMultithread(conf.getInt("myconf.num.multithread", 1));
			}

			// int cur_gpu = running_num_of_gpu_yarnchild;
			String stat = "NodeManager launched\n";
			stat += "max_gpu : " + GPU.getMaxChild() + "\n";
			stat += "min_gpu : " + GPU.getMinChild() + "\n";
			stat += "max_cpu : " + CPU.getMaxChild() + "\n";
			stat += "use_multithread : " + CPU.getUseMultithread() + "\n";
			stat += "num_multithread : " + CPU.getNumMultithread() + "\n";
			stat += "max_container  : " + AM.getMaxSlot() + "\n";
			stat += "cpugpu_proportion : + " + cpugpu_proportion + "\n";
			stat += "debug_listener_address : " + debug_listener_address + "\n";
			stat += "bUse_debug_listener : " + bUse_debug_listener + "\n";
			stat += "bUse_dynamic_scheduler : " + bUse_dynamic_scheduler + "\n";
			if (this.bUse_dynamic_scheduler && bUse_debug_listener) {
				if (max_start_mode) {
					stat += " ---- scheduler mode : max_start\n";
				} else {
					stat += " ---- scheduler mode : conservative \n";
				}
			}
			stat += "upper_threshold : " + upper_threshold + "\n";
			stat += "gpu_threshold : " + gpu_threshold + "\n";
			stat += "load balancing policy : " + lbp + "\n";
			stat += "matmul_time_threshold : " + matmul_time_threshold + "\n";
			stat += "matmul_count_threshold : " + matmul_count_threshold + "\n";
			stat += "matmul_length : " + matmul_length + "\n";
			stat += "time to monitor : " + lbp + "\n";

			if (expr_title.length() > 0) {
				stat += "experiment title : " + expr_title + "\n";
			}

			//stat += "profile gpu util : "+ gpumonitor.getProfiledGpuUtil() +"\n";
			gpumonitor.sendMsgSafe(stat);

			super.serviceInit(conf);
		}

	@Override
		protected void serviceStop() throws Exception {
			containerLauncher.shutdownNow();
			gpumonitor.onStop();
			super.serviceStop();
		}

	/********************************************************************
	 * AUXILIARY FUNCTIONS - LAUNCH AND CEANUP YARN CHILD
	 ********************************************************************/
	private float calculateP(ChildMaster CPU, ChildMaster GPU) {
		float p = cpugpu_proportion;
		float c = cpugpu_proportion, g = 1;

		if (CPU.getFinished() < CPU.getThreshold()) {
			p = cpugpu_proportion;
		} else if (CPU.getFinished() < CPU.latest_n) {
			c = (float) CPU.getRunningTime() / CPU.getFinished();
		} else {
			c = 0;
			for (int i=0; i<CPU.latest_n; i++) c += CPU.n_running_time[i];
			c /= CPU.latest_n;
		}

		if (GPU.getFinished() < GPU.getThreshold()) {
			p = cpugpu_proportion;
		} else if (GPU.getFinished() < GPU.latest_n) {
			g = (float) GPU.getRunningTime() / GPU.getFinished();
		} else {
			g = 0;
			for (int i=0; i<GPU.latest_n; i++) g += GPU.n_running_time[i];
			g /= GPU.latest_n;
		}

		try {
			if (CPU.getFinished() >= CPU.getThreshold() && GPU.getFinished() >= GPU.getThreshold()) {
				p = c / g;
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return p;
	}

	private int getCurTasks(String mycmd) {
		String s_cur_tasks = (String)mycmd.subSequence(mycmd.indexOf("attempt_")+29, mycmd.indexOf("attempt_")+35);
		return Integer.parseInt(s_cur_tasks);
	}

	private String insert(String mycmd, int pos_insert, String str) {
		//	mycmd = "/home/hylo/local/cuda/cuda-7.5/bin/nvprof --concurrent-kernels on --profile-api-trace all --profile-from-start on --system-profiling off --unified-memory-profiling per-process-device --profile-child-processes -o /home/hylo/api_%p.log " 
		mycmd = mycmd.substring(0, pos_insert)
			+ str
			+ mycmd.substring(pos_insert, mycmd.length());
		return mycmd;
	}

	/********************************************************************
	 *	                   CORE SCHEDULING LOGIC
	 ********************************************************************/
	@Override
		public void handle(ContainersLauncherEvent event) { // TODO: ContainersLauncher launches containers one by one!!

			int cur_tasks = -1;
			String temp1="";

			Container container = event.getContainer();
			ContainerId containerId = container.getContainerId();

			List<String> commands = container.getLaunchContext().getCommands();
			boolean isMRAppMaster = false;
			for (String command : commands) {
				isMRAppMaster = command.contains("MRAppMaster");
			}

			//hyonzin: check that it is map task
			String mycmd = commands.get(0);
			TaskType task_type;
			switch (mycmd.charAt(mycmd.indexOf("attempt")+27)) {
				case 'm':
					task_type = TaskType.MAP;
					break;
				case 'r':
					task_type = TaskType.REDUCE;
					break;
				default:
					task_type = TaskType.UNKNOWN;
					break;
			}


			switch (event.getType()) {
				/********************************************************************
				 *	             LAUNCH CONTAINER
				 ********************************************************************/
				case LAUNCH_CONTAINER:
					Application app = context.getApplications().get(
							containerId.getApplicationAttemptId().getApplicationId());

					ContainerLaunch launch = new ContainerLaunch(context, getConfig(),
							dispatcher, exec, app, event.getContainer(), dirsHandler,
							containerManager);

					// ////////////////////////////////////////
					// yildbs added
					if (isMRAppMaster == false && task_type == TaskType.MAP) 
					{
						gpumonitor.GpuMapPhaseIsRunning();

						AM.newChild();
						int pos_insert = 0;
						pos_insert = mycmd.indexOf("java") + 4;

						if (pos_insert != 0) 
						{
							AM.setNumMapTask(mycmd);
							cur_tasks = getCurTasks(mycmd);

							int N = AM.getNumMapTask() / num_of_nodes;

							if( bOnlyCPU ) 
							{
								mycmd = insert(mycmd, pos_insert, " -DuseGPU=false -DuseMultithread=" + CPU.getUseMultithread() + " -DnumMultithread=" + CPU.getNumMultithread() + " -DR=" + R);
								CPU.willLaunch(containerId);
							}
							else if (lbp == LOAD_BALANCING_POLICY.auto)
							{ 
								if ( isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(),  true) ) 
								{
									mycmd = insert(mycmd, pos_insert, " -DuseGPU=true -DR=" + R);
									GPU.willLaunch(containerId);
									isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);
								}
								else if( bMustGPU )
								{
									//Modified for informing information about mustGPU flag to YarnChild
									mycmd = insert(mycmd, pos_insert, " -DuseGPU=true -DmustGPU=true -DR=" + R);
									GPU.willLaunch(containerId);
									isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);
								}
								else
								{
									float ng, nc;

									if (this.bUse_dynamic_scheduler) 
										ng = gpumonitor.getDesiredNG();  // ng = gpumonitor.X;
									else
										ng = GPU.getMaxChild();

									nc = AM.getMaxSlot() - ng - AM.getMasterNum();
									int a = CPU.getCount();
									float P = calculateP(CPU, GPU);

									float f_n_gpu_iteration = (N-a-1)/P/ng;
									float f_n_cpu_iteration = (nc>0)? ((a+1) / nc) : 0f;

									// eq.4 in the paper
									int n_gpu_iteration = (int) Math.ceil( f_n_gpu_iteration );
									int n_cpu_iteration = (int) Math.ceil( f_n_cpu_iteration );

									temp1 += "@@@@@@@@ ceiling \n";
									temp1 += "NG : " + ng + "\n";
									temp1 += "NC : " + nc + "\n";
									temp1 += "N : " + N + "\n";
									temp1 += "n nodes : " + num_of_nodes + "\n";
									temp1 += "n_gpu_iteration : " + n_gpu_iteration+ "\n";
									temp1 += "n_cpu_iteration : " + n_cpu_iteration+ "\n";
									temp1 += "f_n_gpu_iteration : " + f_n_gpu_iteration+ "\n";
									temp1 += "f_n_cpu_iteration : " + f_n_cpu_iteration+ "\n";
									temp1 += "count_cpu_yarnchild(a) : " + CPU.getCount() + "\n";
									temp1 += "count_gpu_yarnchild : " + GPU.getCount() + "\n";

									if( n_gpu_iteration <= n_cpu_iteration )
									{
										mycmd = insert(mycmd, pos_insert, " -DuseGPU=true -DR=" + R);
										GPU.willLaunch(containerId);
										isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);

										bMustGPU = true;
										gpumonitor.setTail(true);
										temp1 += "### must be gpu \n";
									}
									else
									{
										mycmd = insert(mycmd, pos_insert, " -DuseGPU=false -DuseMultithread=" + CPU.getUseMultithread() + " -DnumMultithread=" + CPU.getNumMultithread() + " -DR=" + R);
										CPU.willLaunch(containerId);
									}
									temp1 += "@@@@@@@@ ceilingend \n";
								}
							} /* Tail scheduling auto */ 
							else if (lbp == LOAD_BALANCING_POLICY.threshold)
							{
								if( cur_tasks > (int)((float)AM.getNumMapTask()*gpu_threshold))
								{
									temp1 += "@@@@@@@@ not auto ceiling \n";
									mycmd = insert(mycmd, pos_insert, " -DuseGPU=true -DR=" + R);
									GPU.willLaunch(containerId);
									isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);
									temp1 += "##### must be gpu2 \n";
									temp1 += "@@@@@@@@ not auto ceiling end \n";
								}
								else if (isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(),  true) ) 
								{
									mycmd = insert(mycmd, pos_insert, " -DuseGPU=true -DR=" + R);
									GPU.willLaunch(containerId);
									isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);
								}
								else 
								{
									mycmd = insert(mycmd, pos_insert, " -DuseGPU=false -DuseMultithread=" + CPU.getUseMultithread() + " -DnumMultithread=" + CPU.getNumMultithread() + " -DR=" + R);
									CPU.willLaunch(containerId);
								}
							}

							commands.remove(0);
							commands.add(0, mycmd);
						}
					} else if( isMRAppMaster ){ // is MR AppMaster
						AM.newMaster();
						CPU.setMaxChild(AM.getChildNum() - GPU.getMaxChild() - AM.getMasterNum());
					}

					containerLauncher.submit(launch);
					running.put(containerId, launch);
					break;

					/********************************************************************
					 *	             RECOVER CONTAINER
					 ********************************************************************/
				case RECOVER_CONTAINER:
					if( isMRAppMaster == false && task_type == TaskType.MAP ){
						AM.newChild();
					}else if( isMRAppMaster ){
						AM.newMaster();
					}

					app = context.getApplications().get(
							containerId.getApplicationAttemptId().getApplicationId());
					launch = new RecoveredContainerLaunch(context, getConfig(),
							dispatcher, exec, app, event.getContainer(), dirsHandler,
							containerManager);
					containerLauncher.submit(launch);
					running.put(containerId, launch);
					break;

					/********************************************************************
					 *	             CLEANUP CONTAINER
					 ********************************************************************/
				case CLEANUP_CONTAINER:
					if( isMRAppMaster == false && task_type == TaskType.MAP ){
						AM.delChild();

						if (commands.toString().contains("useGPU=true")) {
							GPU.willCleanup(containerId);
							isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);
						}else{
							CPU.willCleanup(containerId);
							if( bUse_debug_listener )
								gpumonitor.CpuMapTaskIsOver();
						}

						//if( (finished_num_of_cpu_yarnchild + finished_num_of_gpu_yarnchild) >= AM.getNumMapTask()) 
						//if (AM.getChildNum() == 0) {

						if( (CPU.getFinished() + GPU.getFinished()) >= AM.getNumMapTask()) {
							CPU.setMaxChild(AM.getMaxSlot() - GPU.getMaxChild());
							bMustGPU = false;

							//			running_num_of_all_yarnchild = 0;
							GPU.clear();
							CPU.clear();

							gpumonitor.GpuMapPhaseIsOver();
							isGpuAvailable(GPU.getRunning(), GPU.getMaxChild(), false);
						}
					}else if( isMRAppMaster ){ // is MP AppMaster
						AM.delMaster();
						gpumonitor.onJobFinished();
					}

					ContainerLaunch launcher = running.remove(containerId);
					if (launcher == null) {
						// Container not launched. So nothing needs to be done.
						return;
					}
					// Cleanup a container whether it is running/killed/completed, so
					// that
					// no sub-processes are alive.
					try {
						launcher.cleanupContainer();
					} catch (IOException e) {
						LOG.warn("Got exception while cleaning container "
								+ containerId + ". Ignoring.");
					}
					break;
						default:
					break;
					} /* event.getType() */

					/********************************************************************
					 * PRINTING INFORMATION
					 ********************************************************************/
					if(task_type==TaskType.MAP)

					{
						String stat = "";
						stat += "commands:" + commands + "\n";
						stat += "Event type:" + event.getType().toString() + "\n";
						stat += "is MR AM ? : " + isMRAppMaster + "\n";
						stat += "running_num_of_mr : " + AM.getMasterNum() + "\n";
						stat += "P : " + calculateP(CPU, GPU) + "\n";
						stat += "cur_cpu : " + CPU.getRunning() + "\n";
						stat += "cur_gpu : " + GPU.getRunning() + "\n";
						stat += "max_gpu : " + GPU.getMaxChild() + "\n";
						stat += "bMustGPU : " + bMustGPU + "\n";
						stat += "bUseMPS : " + this.bUseMps + "\n";
						stat += "max_container  : " + AM.getMaxSlot() + "\n";
						stat += "progress : ( " 
							+ (CPU.getFinished() + GPU.getFinished()) + " ("
							+ (CPU.getFinished() + GPU.getFinished() + CPU.getRunning() + GPU.getRunning())
							+ ") / " + AM.getNumMapTask() + " )\n";
						stat += "finished_cpu  : " + CPU.getFinished() + "\n";
						stat += "finished_gpu  : " + GPU.getFinished() + "\n";
						stat += "load balancing policy : " + lbp + "\n";
						stat += temp1;
						gpumonitor.sendMsgSafe(stat);
					}
			} /* handle() */
		}

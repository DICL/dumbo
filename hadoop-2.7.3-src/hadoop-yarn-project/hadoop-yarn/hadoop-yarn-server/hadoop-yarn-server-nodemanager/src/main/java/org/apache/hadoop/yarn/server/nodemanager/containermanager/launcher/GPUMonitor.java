package org.apache.hadoop.yarn.server.nodemanager.containermanager.launcher;

import java.util.concurrent.Semaphore;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;
import java.net.UnknownHostException;

public class GPUMonitor implements Runnable {
    boolean bIsGpuAvailable = true;
    boolean bRunning = true;
    int desired = 0;
    STATE state = STATE.EMPTY;
    float upper_threshold = 80.0f;
    float lower_threshold = 10.0f;
    int memory;
    int processCount;
    float util_avg;
    int matmul_full_count = 0;
    float matmul_threshold = 8.0f; // hyonzin: if matmul's duration is more than this value,
                                   // GPU is considered to be fully utilizing. (millisecond)
    int matmul_full_count_threshold = 14; // hyonzin: if matmul_full_count equals this value,
                                          // GPU is considered to be fully utilizing.
    int max_num_of_gpu_yarnchild = 8;
    boolean bNumOverMaxGpu = false;
    String debug_listener_address; 
    int debug_listener_port;
    boolean bUse_debug_listener;
    int max_gpu;
    int dynamic_policy = 0;
    int num_min_gpu_maptask = 0;
    boolean print_state=false;
    boolean cpumaptaskisover =false;
    boolean max_start_mode = true;
    public float X = 99999;
    private boolean bMapPhase = false;
    Semaphore sem, sem2;
    Socket s = null;
    PrintWriter out = null;

    public GPUMonitor(String debug_listener_address, int debug_listener_port,
	    boolean bUse_debug_listener, float upper_threshold, Semaphore sem, Semaphore sem2) {
	this.debug_listener_address = debug_listener_address;
	this.debug_listener_port = debug_listener_port;
	this.bUse_debug_listener = bUse_debug_listener;
	this.upper_threshold = upper_threshold;
	this.sem = sem;
	this.sem2 = sem2;	
    }
	
    public void CpuMapTaskIsOver(){
	cpumaptaskisover = true;
    }
    public void SetNumMinGpuMapTask(int num_min_gpu_maptask){
	this.num_min_gpu_maptask = num_min_gpu_maptask;
    }
    public void SetMaxStartMode(boolean max_start_mode){
	this.max_start_mode = max_start_mode;
    }


    void sendMsgSafe(String mes) {
	try {
	    sem2.acquire(1);
	    try {
		sendMsg(mes);
	    } finally {
		sem2.release(1);
	    }
	} catch (Exception e) {
	    e.printStackTrace();
	}
    }

    void sendMsg(String mes) {
	String full_mes = "";
	full_mes += "<<<<<<<<<\n";
	full_mes += mes;
	full_mes += "\n>>>>>\n";
	
	System.out.println("ADDRESS: " + debug_listener_address);
	System.out.println("PORT: " + debug_listener_port);

	try {
	    s = new Socket(debug_listener_address, debug_listener_port);
	    System.out.println("Socket Established!");
	    out = new PrintWriter(s.getOutputStream(), true);
	    out.println(full_mes);
	    s.close();
	} catch (UnknownHostException e) {
	    System.out.println("UnknownHostException!");
	    e.printStackTrace();
	} catch (IOException e) {
	    System.out.println("IOException!");
	    e.printStackTrace();
	}
    }

    public boolean isGPUFull() {
	if( util_avg > upper_threshold
		&& matmul_full_count >= matmul_full_count_threshold ) {
	    return true;
	}
	return false;
    }

    public boolean FSM_conservative(int cur_gpu,int max_gpu) {
	return FSM_conservative(cur_gpu, max_gpu, false);
    }

    public boolean FSM_conservative(int cur_gpu,int max_gpu,boolean isNewMapTask) {
	processCount = cur_gpu;
	this.max_gpu = max_gpu;

	switch (state) {
	    case EMPTY:
		bIsGpuAvailable = true;
		desired = num_min_gpu_maptask;

		if( cur_gpu >= 1 ){
		    //desired = max_gpu;
		    //bNumOverMaxGpu = true;
		    state = STATE.WAIT;
		    bIsGpuAvailable = true;
		}
		break;
	    case INCRE:
		bIsGpuAvailable = true;
		if( cur_gpu >= desired ){
		    state = STATE.WAIT;
		    bIsGpuAvailable = false;
		}else if( cur_gpu==0){
		    state = STATE.EMPTY;
		}
		break;	
	    case WAIT:
		bIsGpuAvailable = false;
		if( cur_gpu < desired ){
		    state = STATE.INCRE;
		}
		if( cpumaptaskisover ){
		    cpumaptaskisover = false;
		    desired++;
		    state = STATE.INCRE;
		}
		if( isGPUFull() ){
		    state = STATE.FULL;
		    bIsGpuAvailable = false;
		}
		break;				

	    case TRANSITION1:
		if( isGPUFull() ){
		    //if( cur_gpu >= desired ){
		    state = STATE.FULL;
		    bIsGpuAvailable = false;
		}else if ( cur_gpu <= desired - 2 ){
		    state = STATE.UNDER;
		    desired = max_gpu;
		    bIsGpuAvailable = true;
		}
		break;
		    case FULL:
		if( !isGPUFull() ){
		    state = STATE.TRANSITION2;
		    bIsGpuAvailable = true;
		    desired = Math.min(cur_gpu+2, max_gpu);
		    X = (desired+cur_gpu)/2;
		}
		break;
		    case TRANSITION2:
		if( desired <= cur_gpu){
		    //state = STATE.TRANSITION3;
		    state = STATE.TRANSITION1;
		    bIsGpuAvailable = false;
		}else if( isGPUFull() ){
		    state = STATE.FULL;
		    bIsGpuAvailable = false;
		}else if ( cur_gpu <= desired - 2 ){
		    state = STATE.UNDER;
		    desired = max_gpu;
		    bIsGpuAvailable = true;
		}
		break;
		    case UNDER:
		if( isGPUFull() ){
		    state = STATE.FULL;
		    bIsGpuAvailable = false;
		}else if( !bMapPhase ){
		    state = STATE.EMPTY;
		    desired = max_gpu;
		    bIsGpuAvailable = true;
		}
		break;
		    default:
		break;
		}

		if( !bMapPhase ){
		    state = STATE.EMPTY;
		    desired = max_gpu;
		    bIsGpuAvailable = true;
		}

		if( bIsGpuAvailable && isNewMapTask ){
		    processCount = processCount + 1;
		}

		return bIsGpuAvailable;
	}

	public boolean FSM_maxStart(int cur_gpu,int max_gpu) {
	    return FSM_maxStart(cur_gpu, max_gpu, false);
	}

	public boolean FSM_maxStart(int cur_gpu,int max_gpu,boolean isNewMapTask) {
	    processCount = cur_gpu;
	    this.max_gpu = max_gpu;

	    switch (state) {
		case EMPTY:
		    bIsGpuAvailable = true;

		    if( cur_gpu >= max_gpu ){
			desired = max_gpu;
			//bNumOverMaxGpu = true;
			bIsGpuAvailable = false;
			state = STATE.TRANSITION1;
		    }
		    break;
		case TRANSITION1:
		    if( isGPUFull() ){
			//if( cur_gpu >= desired ){
			state = STATE.FULL;
			bIsGpuAvailable = false;
		    }else if ( cur_gpu <= desired - 2 ){
			state = STATE.UNDER;
			desired = max_gpu;
			bIsGpuAvailable = true;
		    }
		    break;
			case FULL:
		    if( !isGPUFull() ){
			state = STATE.TRANSITION2;
			bIsGpuAvailable = true;
			desired = Math.min(cur_gpu+2, max_gpu);
			//X = cur_gpu;
			X = (desired+cur_gpu)/2;
		    }
		    break;
			case TRANSITION2:
		    if( desired <= cur_gpu){
			//state = STATE.TRANSITION3;
			state = STATE.TRANSITION1;
			bIsGpuAvailable = false;
		    }else if( isGPUFull() ){
			state = STATE.FULL;
			bIsGpuAvailable = false;
		    }else if ( cur_gpu <= desired - 2 ){
			state = STATE.UNDER;
			desired = max_gpu;
			bIsGpuAvailable = true;
		    }
		    break;
			case UNDER:
		    if( isGPUFull() ){
			state = STATE.FULL;
			bIsGpuAvailable = false;
		    }else if( !bMapPhase ){
			state = STATE.EMPTY;
			desired = max_gpu;
			bIsGpuAvailable = true;
		    }
		    break;
			default:
		    break;
		    }

		    if ( !bMapPhase ){
			state = STATE.EMPTY;
			desired = max_gpu;
			bIsGpuAvailable = true;
		    }

		    if( bIsGpuAvailable && isNewMapTask ){
			processCount = processCount + 1;
		    }

		    return bIsGpuAvailable;
	    }

	    public boolean isGpuAvailable(int cur_gpu, int max_gpu, boolean isNewMapTask) {
		if( max_start_mode )
		    return FSM_maxStart(cur_gpu, max_gpu, isNewMapTask);
		else
		    return FSM_conservative(cur_gpu, max_gpu, isNewMapTask);
	    }

	    enum STATE {
		//INITIAL, TRANSITION1, TRANSITION2, INTERMEDIATE, FULL, EXTRA
		EMPTY, TRANSITION1, FULL,TRANSITION2, TRANSITION3,UNDER,
		WAIT,INCRE
	    }

	    public void run() {
		int[] utilizations = new int[16];
		int utilization_sum = 0;
		float[] matmuls = new float[16];
		int matmul_sum = 0;
		int[] memories = new int[16];
		int idxCount = 0;
		//		int memories_sum = 0;	// not used now
		//		boolean bIsUpper = false;	// not used now

		for (int i = 0; i < utilizations.length; i++) {
		    utilizations[i] = 0;
		    matmuls[i] = 0.0f;
		    memories[i] = 0;
		}

		String err = onStart();
		String mes = "<Java>GPUMonitor : message from JNI : " + err;
		sendMsgSafe(mes);

		long t_pre_send_msg = 0;
		while (bRunning) {
		    long t_total_elapsed = System.nanoTime();
		    long t_elapsed = System.nanoTime();
		    t_elapsed = System.nanoTime() - t_elapsed;

		    int value = getState();
		    int utilization = (value & 0x000000FF);
		    int memory = (value & 0x0000FF00) >> 8;
		    int infoCount = (value & 0x0000FF00) >> 8;
		    float matmul = doDummyJob();

		    if (matmuls[idxCount] > matmul_threshold) matmul_full_count--;
		    if (matmul            > matmul_threshold) matmul_full_count++;
		    
		    utilization_sum += utilization - utilizations[idxCount];
		    utilizations[idxCount] = utilization;
		    matmuls[idxCount] = matmul;
		    idxCount = (idxCount + 1) & 0xF;

		    util_avg = (float) utilization_sum / 16.0f;
		    t_total_elapsed = System.nanoTime() - t_total_elapsed;
		    try {
			sem.acquire(1);
			try {
			    isGpuAvailable( processCount, this.max_gpu , false);
			} finally {
			    sem.release(1);
			}
		    } catch (Exception e) {
			e.printStackTrace();
		    }

		    try {
			Thread.sleep(66);
		    } catch (InterruptedException e) {
			e.printStackTrace();
		    }

		    if( !bUse_debug_listener ) continue;
		    long t_cur = System.currentTimeMillis();
		    if (t_cur - t_pre_send_msg > 1000) {
			t_pre_send_msg = t_cur;
			mes = "";
			if( max_start_mode ){
			    mes += " ---- scheduler mode : max_start\n";
			}else{
			    mes += " ---- scheduler mode : conservative \n";
			}
			mes += "util!!! : ";
			for (int util : utilizations) {
			    mes += "" + util + ",";
			}
			mes += "sum : " + utilization_sum + "\n";
			mes += "state : " + state + "\n";
			mes += "isAvail : " + bIsGpuAvailable + "\n";
			mes += "desired : " + desired + "\n";
			mes += "X : " + X + "\n";
			mes += "processCount : " + processCount + "\n";
			mes += "upper : " + upper_threshold + "\n";

			mes += "utilization : " + utilization + "\n";
			mes += "util_avg : " + util_avg + "\n";
			mes += "gpu memory : " + memory + "\n";
			mes += "gpu infoCount : " + infoCount + "\n";

			mes += "matmul : " + matmul_full_count + "\n";
			
			mes += "elapsed : " + (long) (t_elapsed / 1000) + " us \n";
			mes += "total elapsed : " + (long)(t_total_elapsed/1000) + " us\n";

			if( this.bMapPhase ){	
			    print_state = false;
			    sendMsgSafe(mes);
			} else {
			    if( !print_state ){
				print_state = true;
				sendMsgSafe(mes);
			    }
			} /* if( this.bMapPhase ) */
		    } /* if (t_cur - t_pre_send_msg > 1000) */
		} /* while (bRunning) */
	    } /* run */

	    public native String onStart();
	    public native int getState();
	    public native float doDummyJob();
	    public native void onJobFinished();
	    public native void onStop();

	    static {
		System.loadLibrary("JNI_GPUMonitor");
	    }

	    public void setNumMinGpuMapTask(int num_min_gpu_maptask) {
		this.num_min_gpu_maptask = num_min_gpu_maptask;
	    }

	    public void GpuMapPhaseIsRunning() {
		this.bMapPhase = true;
	    }

	    public void GpuMapPhaseIsOver() {
		this.bMapPhase = false;
	    }

	}

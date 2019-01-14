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
  int delta = 0;
  long pre_wait_time = 0;
  long time_to_wait = 15000; // ms
  long time_to_monitor = 120000; // ms
  STATE state = STATE.EMPTY;
  float upper_threshold;
  int memory;
  int processCount;
  float util_avg;
  int matmul_full_count = 0;

  boolean bUseOriginalScheduler = false;
  boolean bUseNvml = true;
  boolean bUseMatmul = true;
  boolean bUseDynamicScheduler;

  float matmul_threshold; // hyonzin: if matmul's duration is more than this value,
  // GPU is considered to be fully utilizing. (millisecond)
  int matmul_length;
  int matmul_full_count_threshold;

  // Maximum number of GPU tasks(yarnchild)
  int max_num_of_gpu_yarnchild = 8;
  // True if the number of GPU yarnchild is over the max value
  boolean bNumOverMaxGpu = false;
  // Socket for debugging
  String debug_listener_address; 
  int debug_listener_port;
  boolean bUse_debug_listener;

  int min_gpu,  max_gpu;
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
      boolean bUse_debug_listener, float upper_threshold, int matmul_length, float matmul_threshold, int matmul_full_count_threshold, Semaphore sem, Semaphore sem2) {
    this.debug_listener_address = debug_listener_address;
    this.debug_listener_port = debug_listener_port;
    this.bUse_debug_listener = bUse_debug_listener;
    this.upper_threshold = upper_threshold;
    this.matmul_length = matmul_length;
    this.matmul_threshold = matmul_threshold;
    this.matmul_full_count_threshold = matmul_full_count_threshold;
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

  public boolean isGpuFull() {
    return( (!this.bUseNvml || (util_avg > upper_threshold))
        && (!this.bUseMatmul || (matmul_full_count >= matmul_full_count_threshold)) );
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
        if( isGpuFull() ){
          state = STATE.FULL;
          bIsGpuAvailable = false;
        }
        break;        

      case DELAY2INC:
        if( isGpuFull() ){
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
        if( !isGpuFull() ){
          state = STATE.DELAY2DEC;
          bIsGpuAvailable = true;
          desired = Math.min(cur_gpu+2, max_gpu);
          X = (desired+cur_gpu)/2;
        }
        break;
      case DELAY2DEC:
        if( desired <= cur_gpu){
          //state = STATE.TRANSITION3;
          state = STATE.DELAY2INC;
          bIsGpuAvailable = false;
        }else if( isGpuFull() ){
          state = STATE.FULL;
          bIsGpuAvailable = false;
        }else if ( cur_gpu <= desired - 2 ){
          state = STATE.UNDER;
          desired = max_gpu;
          bIsGpuAvailable = true;
        }
        break;
      case UNDER:
        if( isGpuFull() ){
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
            state = STATE.DELAY2INC;
          }
          break;
        case DELAY2INC:
          if( isGpuFull() ){
            //if( cur_gpu >= desired ){
            state = STATE.FULL;
            bIsGpuAvailable = false;
          }else if ( cur_gpu < desired - 1 ){
            state = STATE.UNDER;
            //desired = max_gpu;
            bIsGpuAvailable = true;
          }
          break;
        case FULL:
          if( !isGpuFull() ){
            state = STATE.DELAY2DEC;
            bIsGpuAvailable = true;
            desired = Math.min(cur_gpu+1, max_gpu);
            //X = cur_gpu;
            X = (desired+cur_gpu)/2;
          }
          break;
        case DELAY2DEC:
          if( desired <= cur_gpu){
            //state = STATE.TRANSITION3;
            state = STATE.DELAY2INC;
            bIsGpuAvailable = false;
          }else if( isGpuFull() ){
            state = STATE.FULL;
            bIsGpuAvailable = false;
          }else if ( cur_gpu < desired - 1 ){
            state = STATE.UNDER;
            //desired = max_gpu;
            bIsGpuAvailable = true;
          }
          break;
        case UNDER:
          if( isGpuFull() ){
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
        if (this.bUseOriginalScheduler) {
          if( max_start_mode )
            return FSM_maxStart(cur_gpu, max_gpu, isNewMapTask);
          else
            return FSM_conservative(cur_gpu, max_gpu, isNewMapTask);
        } else {
          return FSM_binary_search(cur_gpu);
        }
      }

      public boolean FSM_binary_search(int cur_gpu) {
        if (!bMapPhase) {
          state = STATE.EMPTY;
        }

        switch (state) {
          case EMPTY:
            desired = max_gpu;
            if( cur_gpu > 0 ){
              delta = Math.max((desired/2), 1);
              state = STATE.DELAY;
            }
            else break;
          case DELAY:
            if( cur_gpu == desired) {
              pre_wait_time = System.currentTimeMillis();
              state = STATE.WAIT;
            }
            else break;
          case WAIT:
            if( System.currentTimeMillis() - pre_wait_time > time_to_wait) {
              pre_wait_time = System.currentTimeMillis();
              state = STATE.MONITOR;
            }
            else break;
          case MONITOR:
            if( System.currentTimeMillis() - pre_wait_time > time_to_monitor) {
              state = STATE.ADJUST;
            }
            else break;
          case ADJUST:
            desired = (isGpuFull())? Math.max(desired-delta, min_gpu) : Math.min(desired+delta, max_gpu);
            delta = Math.max((delta/2), 1);
            state = STATE.DELAY;
            break;
          case TAIL:
            return true;
          default:
            break;
        }

        return (cur_gpu < desired);
      }

      enum STATE {
        //INITIAL, DELAY2INC, DELAY2DEC, INTERMEDIATE, FULL, EXTRA
        EMPTY,
        DELAY2INC, FULL,DELAY2DEC, TRANSITION3,UNDER,
        WAIT,INCRE,
        DELAY, MONITOR, ADJUST, TAIL
      }

      public void run() {
        int[] utilizations = new int[16];
        int utilization_sum = 0;
        float[] matmuls = new float[matmul_length];
        int matmul_sum = 0;
        int[] memories = new int[16];
        int idxCount = 0;
        int matmulIdxCount = 0;
        int measureCount = 0;
        //    int memories_sum = 0;  // not used now
        //    boolean bIsUpper = false;  // not used now

        for (int i = 0; i < utilizations.length; i++) {
          utilizations[i] = 0;
        }
        for (int i = 0; i < matmul_length; i++) {
          matmuls[i] = 0.0f;
        }
        for (int i = 0; i < memories.length; i++) {
          memories[i] = 0;
        }

        String err = onStart();
        String mes = "<Java>GPUMonitor : message from JNI : " + err;
        sendMsgSafe(mes);

        long t_pre_send_msg = 0;
        while (bUseDynamicScheduler && bRunning) {
          long t_total_elapsed = System.nanoTime();
          long t_elapsed = System.nanoTime();
          t_elapsed = System.nanoTime() - t_elapsed;

          int value = getState();
          int utilization = (value & 0x000000FF);
          int memory = (value & 0x0000FF00) >> 8;
          int infoCount = (value & 0x0000FF00) >> 8;
          float matmul = doDummyJob();

          if (matmuls[matmulIdxCount] >= matmul_threshold) matmul_full_count--;
          if (matmul                  >= matmul_threshold) matmul_full_count++;
          matmuls[matmulIdxCount] = matmul;

          utilization_sum += utilization - utilizations[idxCount];
          utilizations[idxCount] = utilization;

          idxCount = (idxCount + 1) & 0xF;
          matmulIdxCount = (matmulIdxCount < matmul_length - 1)? matmulIdxCount+1 : 0;

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

          measureCount++;

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
            mes += "measure count : " + measureCount + "\n"; measureCount = 0;
            //mes += "util!!! : ";
            //for (int util : utilizations) {
            //    mes += "" + util + ",";
            //}
            //mes += "sum : " + utilization_sum + "\n";
            mes += "state : " + state + "\n";

            //if (state == STATE.WAIT || state == STATE.MONITOR) {
            //  mes += "<" + (System.currentTimeMillis() - pre_wait_time) + ">\n";
            //}

            mes += "is_gpu_full : " + (isGpuFull()) + "\n";

            //mes += "isAvail : " + bIsGpuAvailable + "\n";
            mes += "desired : " + desired + "\n";
            mes += "delta : " + delta + "\n";
            //mes += "X : " + X + "\n";
            //mes += "processCount : " + processCount + "\n";
            //mes += "upper : " + upper_threshold + "\n";

            //mes += "utilization : " + utilization + "\n";
            mes += "util_avg : " + util_avg + "\n";
            //mes += "gpu memory : " + memory + "\n";
            //mes += "gpu infoCount : " + infoCount + "\n";

            mes += "matmul : " + (int)((double)matmul_full_count / matmul_length * 100) + "\n";

            for (int iu = ((matmulIdxCount-16) > 0)? (matmulIdxCount-16) : 0; iu < matmulIdxCount ; iu++) {
              mes += " " + (Math.round(matmuls[iu]));
            }
            for (int iu = matmul_length + matmulIdxCount - 16; iu < matmul_length ; iu++) {
              mes += " " + (Math.round(matmuls[iu]));
            }
            mes += "\n";

            //mes += "elapsed : " + (long) (t_elapsed / 1000) + " us \n";
            //mes += "total elapsed : " + (long)(t_total_elapsed/1000) + " us\n";

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

      public void setTimeToWait(long t) {
        this.time_to_wait = t;
      }

      public void setTimeToMonitor(long t) {
        this.time_to_monitor = t;
      }

      public long getTimeToWait() {
        return this.time_to_wait;
      }

      public long getTimeToMonitor() {
        return this.time_to_monitor;
      }

      public void setMinGpu(int n) {
        this.min_gpu = n;
      }

      public int getMinGpu() {
        return this.min_gpu;
      }

      public void setMaxGpu(int n) {
        this.max_gpu = n;
      }

      public int getMaxGpu() {
        return this.max_gpu;
      }

      public void setTail(boolean b) {
        if (b) this.state = STATE.TAIL;
        else this.state = STATE.EMPTY;
      }

      public void useOriginalScheduler(boolean b) {
        this.bUseOriginalScheduler = b;
      }

      public void useNvml(boolean b) {
        this.bUseNvml = b;
      }

      public void useMatmul(boolean b) {
        this.bUseMatmul = b;
      }

      public int getDesiredNG() {
        return this.desired;
      }

      public void useDynamicScheduler(boolean b) {
        this.bUseDynamicScheduler = b;
      }
    }

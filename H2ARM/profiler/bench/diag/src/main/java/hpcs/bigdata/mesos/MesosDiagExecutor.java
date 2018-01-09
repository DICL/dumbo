package hpcs.bigdata.mesos;

import org.apache.mesos.*;
import org.apache.mesos.Protos.*;
import java.util.*;
import java.util.regex.*;
import java.io.*;
import java.net.*;
import java.lang.Thread;

public class MesosDiagExecutor implements Executor {

  @Override
  public void registered(ExecutorDriver driver, ExecutorInfo executorInfo,
      FrameworkInfo frameworkInfo, SlaveInfo slaveInfo) {
    System.out.println("Registered executor on " + slaveInfo.getHostname());
  }

  @Override
  public void reregistered(ExecutorDriver driver, SlaveInfo executorInfo) {
  }

  @Override
  public void disconnected(ExecutorDriver driver) {
  }

  @Override
  public void launchTask(ExecutorDriver pDriver, TaskInfo pTaskInfo) {
    // Start task with status running
    TaskStatus status = TaskStatus.newBuilder().setTaskId(pTaskInfo.getTaskId())
        .setState(TaskState.TASK_RUNNING).build();
    pDriver.sendStatusUpdate(status);
    byte[] message = new byte[0];

    try {
        Thread.sleep(20000);
    } catch (InterruptedException e) {
        assert false;
    }

    // Send framework message and mark the task as finished
    pDriver.sendFrameworkMessage(message);
    status = TaskStatus.newBuilder().setTaskId(pTaskInfo.getTaskId())
        .setState(TaskState.TASK_FINISHED).build();
    pDriver.sendStatusUpdate(status);

  }

  @Override
  public void killTask(ExecutorDriver driver, TaskID taskId) {
  }

  @Override
  public void frameworkMessage(ExecutorDriver driver, byte[] data) {
  }

  @Override
  public void shutdown(ExecutorDriver driver) {
  }

  @Override
  public void error(ExecutorDriver driver, String message) {
  }

  public static void main(String[] args) throws Exception {
    MesosExecutorDriver driver = new MesosExecutorDriver(new MesosDiagExecutor());
    System.exit(driver.run() == Status.DRIVER_STOPPED ? 0 : 1);
  }

}

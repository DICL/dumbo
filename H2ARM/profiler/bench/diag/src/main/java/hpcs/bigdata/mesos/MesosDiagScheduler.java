package hpcs.bigdata.mesos;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.mesos.Protos.ExecutorID;
import org.apache.mesos.Protos.ExecutorInfo;
import org.apache.mesos.Protos.FrameworkID;
import org.apache.mesos.Protos.MasterInfo;
import org.apache.mesos.Protos.Offer;
import org.apache.mesos.Protos.OfferID;
import org.apache.mesos.Protos.Resource;
import org.apache.mesos.Protos.SlaveID;
import org.apache.mesos.Protos.TaskID;
import org.apache.mesos.Protos.TaskInfo;
import org.apache.mesos.Protos.TaskState;
import org.apache.mesos.Protos.TaskStatus;
import org.apache.mesos.Protos.Value;
import org.apache.mesos.Scheduler;
import org.apache.mesos.SchedulerDriver;

import com.google.protobuf.ByteString;

public class MesosDiagScheduler implements Scheduler {

  private final ExecutorInfo executorDiag;
  private final int totalTasks;
  private int launchedTasks = 0;
  private int finishedTasks = 0;

  public MesosDiagScheduler(ExecutorInfo executorDiag,
      int totalTasks) {
    this.executorDiag = executorDiag;
    this.totalTasks = totalTasks;

  }

  @Override
  public void registered(SchedulerDriver driver, FrameworkID frameworkId, MasterInfo masterInfo) {
    System.out.println("Registered! ID = " + frameworkId.getValue());
  }

  @Override
  public void reregistered(SchedulerDriver driver, MasterInfo masterInfo) {
  }

  @Override
  public void disconnected(SchedulerDriver driver) {
  }

  @Override
  public void resourceOffers(SchedulerDriver driver, List<Offer> offers) {
    for (Offer offer : offers) {
      List<TaskInfo> tasks = new ArrayList<TaskInfo>();
      if (launchedTasks < totalTasks) {
        TaskID taskId = TaskID.newBuilder().setValue(Integer.toString(launchedTasks++)).build();
        System.out.println("Launching task " + taskId.getValue());
        // Task for crawler
        TaskInfo task = TaskInfo
            .newBuilder()
            .setName("task " + taskId.getValue())
            .setTaskId(taskId)
            .setSlaveId(offer.getSlaveId())
            .addResources(
                Resource.newBuilder().setName("cpus").setType(Value.Type.SCALAR)
                    .setScalar(Value.Scalar.newBuilder().setValue(1)))
            .addResources(
                Resource.newBuilder().setName("mem").setType(Value.Type.SCALAR)
                    .setScalar(Value.Scalar.newBuilder().setValue(128)))
            .setData(ByteString.EMPTY) //TODO
            .setExecutor(
                ExecutorInfo.newBuilder(executorDiag)
                    .addResources(
                        Resource.newBuilder().setName("cpus").setType(Value.Type.SCALAR)
                            .setScalar(Value.Scalar.newBuilder().setValue(0.1)))
                    .addResources(
                        Resource.newBuilder().setName("mem").setType(Value.Type.SCALAR)
                            .setScalar(Value.Scalar.newBuilder().setValue(32))))
            .build();

        System.out.println("Launching task " + taskId.getValue());

        tasks.add(task);
      }
      driver.launchTasks(offer.getId(), tasks);
    }
  }

  @Override
  public void offerRescinded(SchedulerDriver driver, OfferID offerId) {
  }

  @Override
  public void statusUpdate(SchedulerDriver driver, TaskStatus status) {

    if (status.getState() == TaskState.TASK_FINISHED || status.getState() == TaskState.TASK_LOST) {
      System.out.println("Status update: task " + status.getTaskId().getValue()
          + " has completed with state " + status.getState());
      finishedTasks++;
      System.out.println("Finished tasks: " + finishedTasks);
      if (finishedTasks == totalTasks) {
        // Once the total allowed tasks are finished, write the graph
        // file
        //String parentDir = new File(System.getProperty("user.dir")).getParent();
        //GraphWriter graphWriter = new GraphWriter();
        //graphWriter.writeDotFile(parentDir + "/result.dot", urlToFileNameMap, edgeList);
        driver.stop();
      }
    } else {
      System.out.println("Status update: task " + status.getTaskId().getValue() + " is in state "
          + status.getState());
    }
  }

  @Override
  public void frameworkMessage(SchedulerDriver driver, ExecutorID executorId, SlaveID slaveId,
      byte[] data) {
  }

  @Override
  public void slaveLost(SchedulerDriver driver, SlaveID slaveId) {
  }

  @Override
  public void executorLost(SchedulerDriver driver, ExecutorID executorId, SlaveID slaveId,
      int status) {
  }

  @Override
  public void error(SchedulerDriver driver, String message) {
    System.out.println("Error: " + message);
  }

}

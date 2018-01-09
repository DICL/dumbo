package hpcs.bigdata.mesos;

import org.apache.mesos.MesosSchedulerDriver;
import org.apache.mesos.Protos.CommandInfo;
import org.apache.mesos.Protos.Credential;
import org.apache.mesos.Protos.ExecutorID;
import org.apache.mesos.Protos.ExecutorInfo;
import org.apache.mesos.Protos.FrameworkInfo;
import org.apache.mesos.Protos.Status;
import org.apache.mesos.Scheduler;

import com.google.protobuf.ByteString;
import hpcs.bigdata.mesos.MesosDiagScheduler;

public class MesosDiagMain {
  public static void main(String[] args) throws Exception {
    if (args.length < 1 || args.length > 3) {
      usage();
      System.exit(1);
    }

    String path = System.getProperty("user.dir")
        + "/target/diag-0.0.1-jar-with-dependencies.jar";

    CommandInfo.URI uri = CommandInfo.URI.newBuilder().setValue(path).setExtract(false).build();

    String commandDiag = "java -cp diag-0.0.1-jar-with-dependencies.jar hpcs.bigdata.mesos.MesosDiagExecutor";
    CommandInfo commandInfoDiag = CommandInfo.newBuilder().setValue(commandDiag).addUris(uri)
        .build();

    ExecutorInfo executorDiag = ExecutorInfo.newBuilder()
        .setExecutorId(ExecutorID.newBuilder().setValue("MesosDiagExecutor"))
        .setCommand(commandInfoDiag).setName("Diag Executor (Java)").setSource("java").build();

    FrameworkInfo.Builder frameworkBuilder = FrameworkInfo.newBuilder().setFailoverTimeout(120000)
        .setUser("") // Have Mesos fill in
        // the current user.
        .setName("Diag Framework (Java)");

    if (System.getenv("MESOS_CHECKPOINT") != null) {
      System.out.println("Enabling checkpoint for the framework");
      frameworkBuilder.setCheckpoint(true);
    }

    Scheduler scheduler = new MesosDiagScheduler(executorDiag, Integer.parseInt(args[1]));

    MesosSchedulerDriver driver = null;
    if (System.getenv("MESOS_AUTHENTICATE") != null) {
      System.out.println("Enabling authentication for the framework");

      if (System.getenv("DEFAULT_PRINCIPAL") == null) {
        System.err.println("Expecting authentication principal in the environment");
        System.exit(1);
      }

      if (System.getenv("DEFAULT_SECRET") == null) {
        System.err.println("Expecting authentication secret in the environment");
        System.exit(1);
      }

      Credential credential = Credential.newBuilder()
          .setPrincipal(System.getenv("DEFAULT_PRINCIPAL"))
          .setSecret(System.getenv("DEFAULT_SECRET")).build();

      frameworkBuilder.setPrincipal(System.getenv("DEFAULT_PRINCIPAL"));

      driver = new MesosSchedulerDriver(scheduler, frameworkBuilder.build(), args[0], credential);
    } else {
      frameworkBuilder.setPrincipal("test-framework-java");

      driver = new MesosSchedulerDriver(scheduler, frameworkBuilder.build(), args[0]);
    }

    int status = driver.run() == Status.DRIVER_STOPPED ? 0 : 1;

    // Ensure that the driver process terminates.
    driver.stop();

    System.exit(status);
  }

  private static void usage() {
    String name = MesosDiagScheduler.class.getName();
    System.err.println("Usage: " + name + " master <tasks> <url>");
  }

}

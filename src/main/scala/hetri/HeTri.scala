package hetri

import java.io.OutputStreamWriter
import java.net.{InetAddress, ServerSocket}
import java.util.StringTokenizer
import java.util.concurrent.TimeUnit

import akka.actor.{ActorSystem, Props}
import com.typesafe.config.{Config, ConfigFactory}
import hetri.actor._
import hetri.allocator.{GreedyAllocator, MultiLevelColoring, RandomAllocator, TaskAllocator}
import hetri.graph.{CSR, CSRV, Graph}
import org.apache.hadoop.conf.{Configuration, Configured}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapreduce.{Job, Mapper}
import org.apache.hadoop.mapreduce.lib.input.{FileInputFormat, NLineInputFormat}
import org.apache.hadoop.mapreduce.lib.output.NullOutputFormat
import org.apache.hadoop.util.{Tool, ToolRunner}
import org.apache.hadoop.yarn.api.records.NodeState
import org.apache.hadoop.yarn.client.api.YarnClient

import scala.concurrent.Await
import scala.concurrent.duration.Duration

class HeTri extends Configured with Tool {

  var numTriangles: Long = 0

  override def run(args: Array[String]): Int = {
    val conf = getConf

    val input = args(0)
    val part = input + ".part"
    val seed = input + ".seed"

    conf.setLong("mapred.task.timeout", 0L)
    conf.set("part", part)

    val numColors = conf.getInt("numColors", 0)
    val alloc = conf.get("allocator", "mlc") match {
      case "mlc" => new MultiLevelColoring(numColors)
      case "rand" => new RandomAllocator(numColors)
      case "greedy" => new GreedyAllocator(numColors)
    }



    val job = Job.getInstance(conf, "[HeTri]" + input + "," + numColors + "," + conf.get("allocator", "mlc"))
    job.setJarByClass(getClass)

    val workers = getWorkers(job.getConfiguration)

    createSeed(job.getConfiguration, seed, workers)

    val system = netStart(numColors, alloc, job.getConfiguration)

    job.setMapperClass(classOf[HeTri.GTEMapper])
    job.setNumReduceTasks(0)

    job.setInputFormatClass(classOf[NLineInputFormat])
    job.setOutputFormatClass(classOf[NullOutputFormat[Any, Any]])
    FileInputFormat.addInputPath(job, new Path(seed))

    job.waitForCompletion(true)

    numTriangles = job.getCounters.findCounter("GTE", "triangles").getValue

    println(numTriangles)

    system.terminate()

    0
  }

  private def netStart(numColors: Int, alloc: TaskAllocator, conf: Configuration): ActorSystem = {

    val systemName = "GTEActorSystem"
    val hostManagerName = "hostManager"
    val hostname = conf.get("yarn.resourcemanager.hostname", "0.0.0.0")
    val config = HeTri.getAkkaRemoteConfig(hostname)
    val system = ActorSystem(systemName, config)
    val port = config.getInt("akka.remote.netty.tcp.port")



    val hostManager = system.actorOf(Props(new HostManagerActor(numColors, alloc)), "hostManager")

    val hostManagerPathString = s"akka.tcp://$systemName@$hostname:$port/user/$hostManagerName"

    conf.set("hostManagerPath", hostManagerPathString)

    system
  }

  private def createSeed(conf: Configuration, seed: String, workers: Array[String]): Unit = {
    val seedOut = new OutputStreamWriter(FileSystem.get(conf).create(new Path(seed)))

    for(i <- workers.indices){
      seedOut.write(i + "\t" + workers(i) + "\n")
    }

    seedOut.close()
  }

  private def getWorkers(conf: Configuration) = {
    val host = conf.get("yarn.resourcemanager.hostname", "0.0.0.0")

    if (host == "0.0.0.0") Array[String]("localhost")
    else {
      val yarn = YarnClient.createYarnClient

      yarn.init(conf)
      yarn.start()

      var res: Array[String] = null

      try {
        val reports = yarn.getNodeReports(NodeState.RUNNING)
        res = new Array[String](reports.size)

        val it = reports.iterator()
        var i = 0
        while(it.hasNext){
          val report = it.next()
          res(i) = report.getNodeId.getHost
          i += 1
        }

        println(reports.size)
      } catch {
        case ex: Exception =>
          println(ex.getMessage)
      }

      yarn.stop()
      res
    }
  }

}

object HeTri{
  def main(args: Array[String]): Unit = {
    ToolRunner.run(new HeTri, args)
  }

  class GTEMapper extends Mapper[Object, Text, Object, Object]{
    override def map(key: Object, value: Text, context: Mapper[Object, Text, Object, Object]#Context): Unit = {

      val conf = context.getConfiguration
      val fs = FileSystem.get(conf)
      val part = new Path(conf.get("part"))
      val numColors = conf.getInt("numColors", 0)
      //      val numCores = conf.getInt("mapreduce.map.cpu.vcores", 1)
      val numCores = Runtime.getRuntime().availableProcessors()
      val st = new StringTokenizer(value.toString)
      val id = st.nextToken.toInt
      val host = st.nextToken

      val gClass: Class[_ <: Graph] = conf.get("graphFormat", "csrv") match {
        case "csrv" => classOf[CSRV]
        case "csr" => classOf[CSR]
      }

      val hostname = InetAddress.getLocalHost.getHostAddress
      val config = HeTri.getAkkaRemoteConfig(hostname)

      val system = ActorSystem("GTEActorSystem", config)

      val hostManagerPathString = context.getConfiguration.get("hostManagerPath")
      val hostManager = system.actorSelection(hostManagerPathString)
      val graphManager = system.actorOf(Props(new GraphManagingActor(part, fs, gClass)), "graphManager")
      val localManager = system.actorOf(Props(new LocalManagerActor(id, numColors,
        numCores + 3, graphManager, hostManager, context)), "localManager")

      Await.result(system.whenTerminated, Duration(1, TimeUnit.DAYS))

      context.getCounter("GTE", "triangleTime").setValue(TriCntActor.executionTime)
      context.getCounter("GTE", "loadingTime").setValue(GraphLoadingActor.executionTime)

    }
  }

  private def getAkkaRemoteConfig(hostname: String): Config ={
    val base: Config = ConfigFactory.parseString("""
       akka {
         actor {
           provider = remote
         }
         remote {
           enabled-transports = ["akka.remote.netty.tcp"]
           transport-failure-detector {
             heartbeat-interval = 1000s
             acceptable-heartbeat-pause = 6000s
           }
        }
       }
    """)

    val soc = new ServerSocket(0)
    val port = soc.getLocalPort
    soc.close()

    val hostConf = ConfigFactory.parseString("akka.remote.netty.tcp.hostname = " + hostname)
    val portConf = ConfigFactory.parseString("akka.remote.netty.tcp.port = " + port)

    base.withFallback(hostConf).withFallback(portConf)
  }

}